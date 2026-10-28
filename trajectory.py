"""Waypoint-based minimum-snap trajectory generation."""

from __future__ import annotations
import jax
import jax.numpy as jnp
import numpy as np


def _poly7(duration, start, end):
    coefficients = jnp.zeros(8, dtype=jnp.float64)
    coefficients = coefficients.at[:4].set(jnp.array([start[0], start[1], start[2] / 2, start[3] / 6]))
    t = duration
    matrix = jnp.array([
        [t**4, t**5, t**6, t**7],
        [4*t**3, 5*t**4, 6*t**5, 7*t**6],
        [12*t**2, 20*t**3, 30*t**4, 42*t**5],
        [24*t, 60*t**2, 120*t**3, 210*t**4],
    ])
    known = jnp.array([
        coefficients[0] + coefficients[1]*t + coefficients[2]*t**2 + coefficients[3]*t**3,
        coefficients[1] + 2*coefficients[2]*t + 3*coefficients[3]*t**2,
        2*coefficients[2] + 6*coefficients[3]*t,
        6*coefficients[3],
    ])
    return coefficients.at[4:].set(jnp.linalg.solve(matrix, end - known))


class MinimumSnapTrajectory:
    """Piecewise degree-seven trajectory, continuous through jerk at internal knots."""

    def __init__(self, waypoints):
        table = np.asarray(waypoints, dtype=float)
        if table.ndim != 2 or table.shape[1] != 5:
            raise ValueError("WAYPOINTS rows must be [time, x, y, z, yaw_deg].")
        if len(table) < 2 or np.any(np.diff(table[:, 0]) <= 0):
            raise ValueError("Waypoint times must be strictly increasing.")
        self.times = jnp.asarray(table[:, 0])
        values = table[:, 1:5]
        values[:, 3] = np.unwrap(np.deg2rad(values[:, 3]))
        self.values = jnp.asarray(values)
        self.duration = float(table[-1, 0])
        self.coefficients = self._build_coefficients()

    @staticmethod
    def _hermite_matrix(duration):
        eye = jnp.eye(8, dtype=jnp.float64)
        return jax.vmap(lambda boundary: _poly7(duration, boundary[:4], boundary[4:]))(eye).T

    @staticmethod
    def _snap_matrix(duration):
        powers = jnp.arange(8, dtype=jnp.float64)
        factors = powers * (powers - 1) * (powers - 2) * (powers - 3)
        exponent = powers[:, None] + powers[None, :] - 7
        valid = (powers[:, None] >= 4) & (powers[None, :] >= 4)
        return jnp.where(valid, factors[:, None] * factors[None, :] * duration**jnp.maximum(exponent, 1) / jnp.maximum(exponent, 1), 0)

    def _build_coefficients(self):
        durations = jnp.diff(self.times)
        segment_count = len(durations)
        matrices = jax.vmap(self._hermite_matrix)(durations)
        costs = jax.vmap(self._snap_matrix)(durations)
        segment_hessians = jnp.einsum("sji,sjk,skl->sil", matrices, costs, matrices)
        hessian = jnp.zeros((4*(segment_count+1), 4*(segment_count+1)))
        for segment in range(segment_count):
            hessian = hessian.at[4*segment:4*segment+8, 4*segment:4*segment+8].add(segment_hessians[segment])
        free = jnp.array([4*k+d for k in range(1, segment_count) for d in (1, 2, 3)])
        fixed = jnp.array([i for i in range(4*(segment_count+1)) if i not in set(np.asarray(free).tolist())])
        derivatives = jnp.zeros((4*(segment_count+1), 4)).at[0::4].set(self.values)
        if len(free):
            solution = jnp.linalg.solve(hessian[free][:, free], -hessian[free][:, fixed] @ derivatives[fixed])
            derivatives = derivatives.at[free].set(solution)
        boundaries = jnp.stack([derivatives[4*s:4*s+8] for s in range(segment_count)])
        return jnp.einsum("sij,sjk->ski", matrices, boundaries)

    @staticmethod
    def _evaluate(coefficients, time):
        powers = jnp.arange(8, dtype=jnp.float64)
        velocity_coefficients = coefficients[1:] * powers[1:]
        acceleration_coefficients = velocity_coefficients[1:] * powers[1:7]
        return (
            jnp.polyval(coefficients[::-1], time),
            jnp.polyval(velocity_coefficients[::-1], time),
            jnp.polyval(acceleration_coefficients[::-1], time),
        )

    def reference(self, time):
        segment = jnp.clip(jnp.searchsorted(self.times, time, side="right") - 1, 0, len(self.times)-2)
        local_time = jnp.clip(time-self.times[segment], 0, self.times[segment+1]-self.times[segment])
        values = jax.vmap(self._evaluate, in_axes=(0, None))(self.coefficients[segment], local_time)
        position, velocity, acceleration = values
        return jnp.concatenate((position[:3], velocity[:3], acceleration[:3], position[3:4]))


class MinimumJerkTrajectory:
    """Piecewise quintic trajectory that stops at every waypoint."""

    def __init__(self, waypoints):
        table = np.asarray(waypoints, dtype=float)
        if table.ndim != 2 or table.shape[1] != 5:
            raise ValueError("WAYPOINTS rows must be [time, x, y, z, yaw_deg].")
        if len(table) < 2 or np.any(np.diff(table[:, 0]) <= 0):
            raise ValueError("Waypoint times must be strictly increasing.")
        self.times = jnp.asarray(table[:, 0])
        values = table[:, 1:5]
        values[:, 3] = np.unwrap(np.deg2rad(values[:, 3]))
        self.values = jnp.asarray(values)
        self.duration = float(table[-1, 0])

    def reference(self, time):
        segment = jnp.clip(jnp.searchsorted(self.times, time, side="right")-1, 0, len(self.times)-2)
        duration = self.times[segment+1]-self.times[segment]
        normalized_time = jnp.clip((time-self.times[segment])/duration, 0.0, 1.0)

        u = normalized_time
        blend = 10*u**3-15*u**4+6*u**5
        blend_rate = (30*u**2-60*u**3+30*u**4)/duration
        blend_acceleration = (60*u-180*u**2+120*u**3)/duration**2

        delta = self.values[segment+1]-self.values[segment]
        position = self.values[segment]+blend*delta
        velocity = blend_rate*delta
        acceleration = blend_acceleration*delta
        return jnp.concatenate((position[:3], velocity[:3], acceleration[:3], position[3:4]))


class InitialHoverTrajectory:
    """Hold the first waypoint before starting the requested trajectory."""

    def __init__(self, trajectory, hover_time):
        self.trajectory = trajectory
        self.hover_time = float(hover_time)
        self.duration = trajectory.duration + self.hover_time
        self.values = trajectory.values
        self._hover_reference = jnp.concatenate((
            trajectory.values[0, :3],
            jnp.zeros(6),
            trajectory.values[0, 3:4],
        ))

    def reference(self, time):
        return jax.lax.cond(
            time < self.hover_time,
            lambda _: self._hover_reference,
            lambda shifted_time: self.trajectory.reference(shifted_time),
            time-self.hover_time,
        )


def build_trajectory(kind, waypoints, initial_hover_time=0.0):
    if kind == "minimum_snap":
        trajectory = MinimumSnapTrajectory(waypoints)
    elif kind == "minimum_jerk":
        trajectory = MinimumJerkTrajectory(waypoints)
    else:
        raise ValueError(f"Unknown trajectory type: {kind}")
    if initial_hover_time < 0:
        raise ValueError("INITIAL_HOVER_TIME cannot be negative.")
    return InitialHoverTrajectory(trajectory, initial_hover_time) if initial_hover_time > 0 else trajectory
