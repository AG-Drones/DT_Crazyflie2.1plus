"""Data export, figures and 3D animation."""

from pathlib import Path
import jax
import numpy as np


def _numpy(data, key):
    return np.asarray(jax.device_get(data[key]), dtype=float)


def save_text(data, output_directory, run_name):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    time = _numpy(data, "time"); state = _numpy(data, "state"); reference = _numpy(data, "reference")
    euler = np.rad2deg(_numpy(data, "euler")); pwm = _numpy(data, "pwm")
    rpm = _numpy(data, "rpm"); thrust = _numpy(data, "thrust")
    output = np.column_stack((time, state[:, :6], reference[:, :6], euler, np.rad2deg(reference[:, 9]), pwm, rpm, thrust))
    header = "time x y z vx vy vz x_ref y_ref z_ref vx_ref vy_ref vz_ref roll pitch yaw yaw_ref "
    header += "pwm_1 pwm_2 pwm_3 pwm_4 rpm_1 rpm_2 rpm_3 rpm_4 thrust_1 thrust_2 thrust_3 thrust_4"
    path = output_directory/f"{run_name}.txt"
    np.savetxt(path, output, fmt="%.8e", header=header, comments="")
    return path


def _save_tracking_figures(data, output_directory, run_name):
    import matplotlib.pyplot as plt
    time = _numpy(data, "time"); state = _numpy(data, "state"); reference = _numpy(data, "reference")
    euler = np.rad2deg(_numpy(data, "euler")); yaw_ref = np.rad2deg(reference[:, 9])
    specifications = [
        ("position", state[:, :3], reference[:, :3], ("x [m]", "y [m]", "z [m]")),
        ("velocity", state[:, 3:6], reference[:, 3:6], ("vx [m/s]", "vy [m/s]", "vz [m/s]")),
    ]
    for name, actual, target, labels in specifications:
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
        for axis, index in zip(axes, range(3)):
            axis.plot(time, target[:, index], "k--", label="reference")
            axis.plot(time, actual[:, index], color="tab:red", label="simulation")
            axis.set_ylabel(labels[index]); axis.grid(alpha=.25)
        axes[0].legend(); axes[-1].set_xlabel("time [s]"); fig.tight_layout()
        fig.savefig(output_directory/f"{run_name}_{name}.png", dpi=200)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 7))
    for axis, values, label in zip(axes, euler.T, ("roll [deg]", "pitch [deg]", "yaw [deg]")):
        axis.plot(time, values, color="tab:red", label="simulation"); axis.set_ylabel(label); axis.grid(alpha=.25)
    axes[2].plot(time, yaw_ref, "k--", label="reference")
    axes[2].legend(); axes[-1].set_xlabel("time [s]"); fig.tight_layout()
    fig.savefig(output_directory/f"{run_name}_attitude.png", dpi=200)

    return plt


def _save_3d_gif(data, output_directory, run_name, parameters):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    state = _numpy(data, "state"); position = state[:, :3]; velocity = state[:, 3:6]
    reference = _numpy(data, "reference")[:, :3]
    time = _numpy(data, "time"); frame_count = max(2, int((time[-1]-time[0])*parameters.gif_fps))
    indices = np.linspace(0, len(time)-1, frame_count, dtype=int)
    all_points = np.vstack((position, reference)); center = (all_points.min(0)+all_points.max(0))/2
    half_range = max(np.ptp(all_points, axis=0).max()/2*1.15, .25)
    fig = plt.figure(figsize=(7, 6)); axis = fig.add_subplot(111, projection="3d")
    axis.plot(*reference.T, "k--", label="reference"); trail, = axis.plot([], [], [], color="tab:blue", label="simulation")
    body, = axis.plot([], [], [], color="tab:blue", lw=3)
    velocity_arrow = None
    axis.set(xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
    axis.set_xlim(center[0]-half_range, center[0]+half_range); axis.set_ylim(center[1]-half_range, center[1]+half_range)
    axis.set_zlim(center[2]-half_range, center[2]+half_range); axis.legend()

    def update(frame):
        nonlocal velocity_arrow
        index = indices[frame]; trail.set_data(position[:index+1, 0], position[:index+1, 1]); trail.set_3d_properties(position[:index+1, 2])
        x, y, z = position[index]; arm = parameters.arm_length
        body.set_data(np.array([x-arm, x+arm, np.nan, x, x]), np.array([y, y, np.nan, y-arm, y+arm]))
        body.set_3d_properties(np.array([z, z, np.nan, z, z]))
        if velocity_arrow is not None:
            velocity_arrow.remove()
        vx, vy, vz = velocity[index]
        velocity_arrow = axis.quiver(
            x, y, z, vx, vy, vz,
            length=0.20,
            normalize=False,
            color="red",
            linewidth=2.5,
            arrow_length_ratio=0.65,
        )
        axis.set_title(f"t = {time[index]:.2f} s")
        return trail, body, velocity_arrow

    animation = FuncAnimation(fig, update, frames=len(indices), interval=1000/parameters.gif_fps, blit=False)
    path = output_directory/f"{run_name}.gif"; animation.save(path, writer=PillowWriter(fps=parameters.gif_fps)); plt.close(fig)


def process_results(data, output_directory, run_name, show_and_save_figs, show_and_save_gif, parameters):
    output_directory = Path(output_directory); text_path = save_text(data, output_directory, run_name)
    if show_and_save_figs:
        output_directory.mkdir(parents=True, exist_ok=True)
        plt = _save_tracking_figures(data, output_directory, run_name)
        plt.show()
    if show_and_save_gif:
        output_directory.mkdir(parents=True, exist_ok=True)
        _save_3d_gif(data, output_directory, run_name, parameters)
    return text_path
