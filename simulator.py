"""Nonlinear six-degree-of-freedom Crazyflie simulator."""

from __future__ import annotations
import jax
from jax import lax
import jax.numpy as jnp
from controller import euler_from_quaternion, rotation_from_quaternion


def _quaternion_rate(q, omega):
    w, x, y, z = q
    p, r, s = omega
    return .5*jnp.array([-x*p-y*r-z*s, w*p+y*s-z*r, w*r+x*s-z*p, w*s+x*r-y*p])


class Simulator:
    def __init__(self, parameters, controller, trajectory):
        self.p = parameters
        self.controller = controller
        self.trajectory = trajectory
        self.inertia_inverse = jnp.linalg.inv(parameters.inertia)

    def rpm_from_pwm(self, pwm):
        return self.p.pwm_to_rpm_slope*pwm + self.p.pwm_to_rpm_intercept

    def translational_drag(self, velocity):
        """Return world-frame linear drag with direction-dependent vertical coefficients."""
        vertical_coefficient = jnp.where(velocity[2] >= 0.0, self.p.drag_z_up, self.p.drag_z_down)
        coefficients = jnp.array([self.p.drag_xy, self.p.drag_xy, vertical_coefficient])
        return -coefficients*velocity

    def derivatives(self, state, pwm, time):
        rotation = rotation_from_quaternion(state[6:10])
        rpm = state[13:17]
        rpm_command = self.rpm_from_pwm(pwm)
        rpm_rate = (rpm_command-rpm)/self.p.tau_rpm
        omega = rpm*2*jnp.pi/60
        thrust = self.p.thrust_coefficient*omega**2
        yaw_torque = self.p.torque_coefficient*omega**2*jnp.array([1,-1,1,-1])
        moments = jnp.array([
            .5*self.p.motor_distance*(-thrust[0]-thrust[1]+thrust[2]+thrust[3]),
            .5*self.p.motor_distance*(-thrust[0]+thrust[1]+thrust[2]-thrust[3]),
            jnp.sum(yaw_torque),
        ])
        force = rotation@jnp.array([0., 0., jnp.sum(thrust)]) + self.translational_drag(state[3:6])
        acceleration = force/self.p.mass + jnp.array([0.,0.,-self.p.gravity])
        rates = state[10:13]
        angular_acceleration = self.inertia_inverse@(moments-jnp.cross(rates, self.p.inertia@rates))
        derivative = jnp.concatenate((state[3:6], acceleration, _quaternion_rate(state[6:10], rates), angular_acceleration, rpm_rate))
        return derivative, rpm, thrust

    def _step_dynamics(self, state, pwm, time):
        dt = self.p.simulation_dt
        f = lambda x, t: self.derivatives(x, pwm, t)[0]
        k1 = f(state, time); k2 = f(state+.5*dt*k1, time+.5*dt)
        k3 = f(state+.5*dt*k2, time+.5*dt); k4 = f(state+dt*k3, time+dt)
        next_state = state + dt*(k1+2*k2+2*k3+k4)/6
        next_state = next_state.at[6:10].set(next_state[6:10]/jnp.linalg.norm(next_state[6:10]))
        return next_state.at[13:17].set(jnp.clip(next_state[13:17], self.rpm_from_pwm(self.p.pwm_min), self.rpm_from_pwm(self.p.pwm_max)))

    def run(self):
        times = jnp.arange(0, self.trajectory.duration+self.p.simulation_dt, self.p.simulation_dt)
        initial_position = self.trajectory.values[0, :3]
        initial_yaw = self.trajectory.values[0, 3]
        initial_rpm = jnp.full(4, self.rpm_from_pwm(self.controller.hover_pwm))
        initial_state = jnp.concatenate((initial_position, jnp.zeros(3), jnp.array([jnp.cos(initial_yaw/2),0,0,jnp.sin(initial_yaw/2)]), jnp.zeros(3), initial_rpm))

        def step(carry, time):
            state, controller_state = carry
            reference = self.trajectory.reference(time)
            pwm, controller_state = self.controller.compute(state, reference, controller_state)
            derivatives, rpm, thrust = self.derivatives(state, pwm, time)
            output = (state, reference, pwm, rpm, thrust, derivatives[3:6], euler_from_quaternion(state[6:10]))
            return (self._step_dynamics(state, pwm, time), controller_state), output

        (_, _), output = lax.scan(step, (initial_state, self.controller.initial_state()), times)
        keys = ("state", "reference", "pwm", "rpm", "thrust", "acceleration", "euler")
        data = dict(zip(keys, output)); data["time"] = times
        return data
