"""Simulation-step scheduling shared by GUI, collection and RL."""
import math

CONTROL_DT = 1 / 200


def control_substeps(model, control_dt=CONTROL_DT):
    ratio = control_dt / model.opt.timestep
    if not math.isfinite(ratio) or ratio < 1 or not math.isclose(ratio, round(ratio), abs_tol=1e-9):
        raise ValueError('Physics timestep must divide the control interval exactly')
    return int(round(ratio))
