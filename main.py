"""Run a Crazyflie digital-twin simulation."""

from pathlib import Path
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

from controller import CrazyflieController
from params import PARAMS
from postprocessing import process_results
from simulator import Simulator
from trajectory import build_trajectory


# -----------------------------------------------------------------------------
# USER SETTINGS
# -----------------------------------------------------------------------------

RUN_NAME = "minimum_snap_demo"

# Each row is [time (s), x (m), y (m), z (m), yaw (deg)].
WAYPOINTS = [
    [0.0, 0.0, 0.0, 1.0, 0.0],
    [4.0, 0.5, 0.0, 1.2, 0.0],
    [8.0, 0.5, 0.5, 0.8, 0.0],
    [12.0, 0.0, 0.0, 1.0, 0.0],
]

# Available options: "minimum_snap" and "minimum_jerk".
TRAJECTORY = "minimum_snap"
INITIAL_HOVER_TIME = 2.0

# The TXT file is always saved. Figures and the GIF are controlled independently.
SHOW_AND_SAVE_FIGS = 1
SHOW_AND_SAVE_GIF = 1
OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "output"

# -----------------------------------------------------------------------------


def main():
    trajectory = build_trajectory(TRAJECTORY, WAYPOINTS, INITIAL_HOVER_TIME)
    controller = CrazyflieController(PARAMS)
    simulator = Simulator(PARAMS, controller, trajectory)
    data = simulator.run()
    path = process_results(
        data,
        OUTPUT_DIRECTORY,
        RUN_NAME,
        SHOW_AND_SAVE_FIGS,
        SHOW_AND_SAVE_GIF,
        PARAMS,
    )
    print(f"Simulation complete. Data saved to: {path}")


if __name__ == "__main__":
    main()
