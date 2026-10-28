# DT Crazyflie

Compact Python digital twin of the Crazyflie 2.1+, based on the architecture and parameters reported by Richardson et al. (2025).

## Run

Install the packages in `requirements.txt`, edit the user settings at the top of `main.py`, then run:

```bash
python main.py
```

Waypoint rows contain time, x, y, z and yaw. Their times are relative to the start of the maneuver. `INITIAL_HOVER_TIME` holds the initial pose before the maneuver begins. Two trajectory keywords are available:

- `minimum_snap`: seventh-order fly-through trajectory, continuous through jerk.
- `minimum_jerk`: quintic point-to-point trajectory with zero velocity and acceleration at every waypoint.

Differences wrt Richardson et al. (2025):
The simulator uses a constant nominal motor model without battery discharge. It always exports a text file. `SHOW_AND_SAVE_FIGS` controls the tracking figures, while `SHOW_AND_SAVE_GIF` independently controls the 3D animation.
Motor speed follows the PWM command through the first-order time constant `tau_rpm`. Optional linear translational drag coefficients are available in `params.py`: one shared horizontal coefficient and separate vertical coefficients for upward and downward motion. They are zero by default.

## Reference

Richardson et al., "Development of a high-fidelity simulation environment for a Crazyflie2.1 quadcopter," MATEC Web of Conferences 417, 04007 (2025). https://doi.org/10.1051/matecconf/202541704007
