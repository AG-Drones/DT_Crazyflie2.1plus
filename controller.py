"""Cascaded Crazyflie PID controller described by Richardson et al. (2025)."""

from typing import NamedTuple
import jax
import jax.numpy as jnp


class ControllerState(NamedTuple):
    integral: jnp.ndarray
    previous: jnp.ndarray
    initialized: jnp.ndarray
    slow_command: jnp.ndarray
    filtered_rates: jnp.ndarray
    tick: jnp.ndarray


GAINS = jnp.array([
    [2, 0, 0], [2, 0, 0], [2, .5, 0],
    [25, 1, 0], [25, 1, 0], [25, 15, 0],
    [6, 3, 0], [6, 3, 0], [6, 1, .35],
    [250, 500, 2.5], [250, 500, 2.5], [120, 16.67, 0],
], dtype=jnp.float64)
INTEGRAL_LIMITS = jnp.array([2,2,2, 2,2,2, 20,20,20, 33.3,33.3,166.7])
OUTPUT_LIMITS = jnp.array([1.1,1.1,1.1, 20,20,20, 360,360,360, 20000,20000,20000])


def _pid(index, target, measurement, dt, state):
    error = target - measurement
    integral = jnp.clip(state.integral[index] + error*dt, -INTEGRAL_LIMITS[index], INTEGRAL_LIMITS[index])
    derivative = jnp.where(state.initialized[index], (measurement-state.previous[index])/dt, 0.0)
    output = GAINS[index,0]*error + GAINS[index,1]*integral - GAINS[index,2]*derivative
    saturated = jnp.clip(output, -OUTPUT_LIMITS[index], OUTPUT_LIMITS[index])
    integral = jnp.where((output != saturated) & (jnp.sign(error) == jnp.sign(output)), state.integral[index], integral)
    state = state._replace(
        integral=state.integral.at[index].set(integral),
        previous=state.previous.at[index].set(measurement),
        initialized=state.initialized.at[index].set(True),
    )
    return saturated, state


class CrazyflieController:
    def __init__(self, parameters):
        self.p = parameters
        omega_hover = jnp.sqrt(parameters.mass*parameters.gravity/(4*parameters.thrust_coefficient))
        rpm_hover = omega_hover*60/(2*jnp.pi)
        self.hover_pwm = (rpm_hover-parameters.pwm_to_rpm_intercept)/parameters.pwm_to_rpm_slope
        self.slow_ratio = int(round(parameters.position_dt/parameters.attitude_dt))

    @staticmethod
    def initial_state():
        return ControllerState(jnp.zeros(12), jnp.zeros(12), jnp.zeros(12, dtype=bool), jnp.zeros(3), jnp.zeros(3), jnp.array(0))

    def _slow_loop(self, state_vector, reference, state):
        yaw = euler_from_quaternion(state_vector[6:10])[2]
        rotation = jnp.array([[jnp.cos(yaw), jnp.sin(yaw)], [-jnp.sin(yaw), jnp.cos(yaw)]])
        position_error = rotation @ (reference[:2]-state_vector[:2])
        velocity_body = rotation @ state_vector[3:5]
        vx, state = _pid(0, position_error[0], 0, self.p.position_dt, state)
        vy, state = _pid(1, position_error[1], 0, self.p.position_dt, state)
        vz, state = _pid(2, reference[2], state_vector[2], self.p.position_dt, state)
        pitch, state = _pid(3, vx, velocity_body[0], self.p.position_dt, state)
        roll_raw, state = _pid(4, vy, velocity_body[1], self.p.position_dt, state)
        thrust_delta, state = _pid(5, vz, state_vector[5], self.p.position_dt, state)
        command = jnp.array([-roll_raw, pitch, thrust_delta])
        return state._replace(slow_command=command)

    def compute(self, state_vector, reference, state):
        state = jax.lax.cond(state.tick % self.slow_ratio == 0, self._slow_loop, lambda _x, _r, s: s, state_vector, reference, state)
        roll, pitch, yaw = jnp.rad2deg(euler_from_quaternion(state_vector[6:10]))
        rates = jnp.rad2deg(state_vector[10:13])
        alpha = jnp.exp(-2*jnp.pi*116.0*self.p.attitude_dt)
        filtered_rates = alpha*state.filtered_rates + (1-alpha)*rates
        state = state._replace(filtered_rates=filtered_rates)
        roll_target, pitch_target, thrust_delta = state.slow_command
        yaw_target = jnp.rad2deg(reference[9])
        roll_rate, state = _pid(6, roll_target, roll, self.p.attitude_dt, state)
        pitch_rate, state = _pid(7, pitch_target, pitch, self.p.attitude_dt, state)
        yaw_rate, state = _pid(8, yaw_target, yaw, self.p.attitude_dt, state)
        roll_pwm, state = _pid(9, roll_rate, filtered_rates[0], self.p.attitude_dt, state)
        pitch_pwm, state = _pid(10, pitch_rate, filtered_rates[1], self.p.attitude_dt, state)
        yaw_pwm, state = _pid(11, yaw_rate, filtered_rates[2], self.p.attitude_dt, state)
        thrust = jnp.clip(self.hover_pwm+1000*thrust_delta, self.p.pwm_min, self.p.pwm_max)
        r, p = .5*roll_pwm, .5*pitch_pwm
        pwm = jnp.array([thrust-r-p+yaw_pwm, thrust-r+p-yaw_pwm, thrust+r+p+yaw_pwm, thrust+r-p-yaw_pwm])
        pwm = jnp.round(jnp.clip(pwm, self.p.pwm_min, self.p.pwm_max))
        return pwm, state._replace(tick=state.tick+1)


def rotation_from_quaternion(q):
    w, x, y, z = q/jnp.linalg.norm(q)
    return jnp.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def euler_from_quaternion(q):
    r = rotation_from_quaternion(q)
    return jnp.array([jnp.arctan2(r[2,1], r[2,2]), jnp.arcsin(jnp.clip(-r[2,0], -1, 1)), jnp.arctan2(r[1,0], r[0,0])])
