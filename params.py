"""Fixed Crazyflie 2.1 parameters from Richardson et al. (2025)."""

from dataclasses import dataclass, field
import jax.numpy as jnp


@dataclass(frozen=True)
class CrazyflieParameters:
    # Rigid-body model
    mass: float = 0.032
    gravity: float = 9.81
    inertia: jnp.ndarray = field(default_factory=lambda: 1e-6 * jnp.array([
        [16.57171, 0.830806, 0.718277],
        [0.830806, 16.55602, 1.800197],
        [0.718277, 1.800197, 29.26165],
    ]))
    motor_distance: float = 0.065
    thrust_coefficient: float = 2.879933e-8
    torque_coefficient: float = 7.23850e-10

    # Motor command
    pwm_min: float = 0.0
    pwm_max: float = 65535.0
    pwm_to_rpm_slope: float = 0.2685
    pwm_to_rpm_intercept: float = 4070.3
    tau_rpm: float = 0.02

    # Linear translational drag [N/(m/s)]
    drag_xy: float = 0.0
    drag_z_up: float = 0.0
    drag_z_down: float = 0.0

    # Controller timing and limits
    simulation_dt: float = 0.002
    position_dt: float = 0.010
    attitude_dt: float = 0.002
    velocity_limit: float = 1.1
    tilt_limit_deg: float = 20.0

    # Plot and animation geometry
    body_length: float = 0.060
    body_width: float = 0.060
    body_height: float = 0.015
    arm_length: float = 0.046
    rotor_radius: float = 0.023
    gif_fps: int = 15


PARAMS = CrazyflieParameters()
