"""Compare independent LQR/NN rollouts with identical deterministic scenarios."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import mujoco
import numpy as np

from control_experiments import (CONTROL_DT, FALL_PITCH, SCENE, Scenario,
                                 control_input, lqr_reference, prepare)
from nn_model import DEFAULT_MODEL, NNPredictor
from robot_lqr import MAX_MOTOR_VEL, RobotLqr, WHEEL_RADIUS
from robot_nn import RobotNN


def evaluate(model_path=DEFAULT_MODEL, duration=10.0):
    if duration <= 0:
        raise ValueError('Duration must be positive')
    predictor = NNPredictor(model_path)
    scenarios = [
        Scenario('balance-positive', .12, duration=duration),
        Scenario('balance-negative', -.12, heading=.7, duration=duration),
        Scenario('gentle-forward', .08, targets=(0, 0, .15, .3, .3), duration=duration, ramp=True),
        Scenario('gentle-reverse', -.08, targets=(0, 0, -.15, -.3, -.3), duration=duration, ramp=True),
        Scenario('forward', .08, targets=(.3,), duration=duration),
        Scenario('reverse', -.08, targets=(-.3,), duration=duration),
        Scenario('forward-reverse', .15, targets=(0, .6, -.6, 0), duration=duration),
        Scenario('turning', -.15, heading=-.8, targets=(.2, -.2), yaw=1, duration=duration),
        Scenario('fast-forward', .05, targets=(0, 2, 4), duration=duration),
        Scenario('fast-reverse', -.05, targets=(0, -2, -4), duration=duration),
    ]
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    rows = []
    for scenario in scenarios:
        for name in ('LQR', 'Neural Network'):
            data = mujoco.MjData(model)
            robot = RobotLqr(model, data) if name == 'LQR' else RobotNN(model, data, predictor)
            substeps = prepare(model, data, robot, scenario)
            pitches, velocity_errors, output_errors = [], [], []
            balancing_time = duration
            fell = False
            for tick in range(round(duration / CONTROL_DT)):
                target = scenario.target_at(tick * CONTROL_DT)
                robot.set_velocity_linear_set_point(target)
                robot.set_yaw(scenario.yaw)
                robot.update_motor_speed()
                inputs = control_input(robot)
                # Compare outputs on the SAME sampled state/filters, not two
                # diverging trajectories. Neither call advances controller filters.
                nn_output = np.clip(predictor.predict(inputs), -MAX_MOTOR_VEL, MAX_MOTOR_VEL)
                output_errors.append(float(nn_output - lqr_reference(inputs)))
                velocity_errors.append(target - robot.control_status.average_wheel_velocity * WHEEL_RADIUS)
                mujoco.mj_step(model, data, nstep=substeps)
                pitch = abs(robot.get_pitch())
                pitches.append(pitch)
                if data.warning.number.any() or not np.isfinite(data.qpos).all():
                    raise RuntimeError(f'Invalid physics in {scenario.name}/{name}')
                if pitch > FALL_PITCH:
                    balancing_time = float(data.time)
                    fell = True
                    break
            rows.append(dict(scenario=scenario.name, controller=name, fell=fell,
                             balancing_time_s=balancing_time,
                             rms_pitch_deg=float(np.rad2deg(np.sqrt(np.mean(np.square(pitches))))),
                             maximum_pitch_deg=float(np.rad2deg(max(pitches))),
                             velocity_tracking_rmse_m_s=float(np.sqrt(np.mean(np.square(velocity_errors)))),
                             output_rmse_rad_s=float(np.sqrt(np.mean(np.square(output_errors)))),
                             output_max_error_rad_s=float(np.max(np.abs(output_errors))),
                             samples=len(pitches)))
    metrics = ('balancing_time_s', 'rms_pitch_deg', 'maximum_pitch_deg',
               'velocity_tracking_rmse_m_s', 'output_rmse_rad_s')
    summary = {}
    for name in ('LQR', 'Neural Network'):
        selected = [r for r in rows if r['controller'] == name]
        summary[name] = {key: float(np.mean([r[key] for r in selected])) for key in metrics}
        summary[name]['maximum_pitch_deg'] = max(r['maximum_pitch_deg'] for r in selected)
        summary[name]['falls'] = sum(r['fell'] for r in selected)
    return dict(model=str(model_path), duration_s=duration, control_hz=1 / CONTROL_DT,
                versions=dict(mujoco=mujoco.__version__, numpy=np.__version__),
                physics_timestep_s=float(model.opt.timestep), fall_threshold_deg=45,
                definitions={'balancing_time': 'time to first |pitch| >45deg, capped at duration',
                             'pitch': 'post-step body pitch until fall, degrees',
                             'velocity': 'target minus raw forward wheel average * radius, RMSE until fall',
                             'output': 'clipped pre-yaw NN minus LQR on identical inputs along each trajectory',
                             'summary': 'unweighted mean across scenarios, except maximum_pitch is the global maximum'},
                scenarios=[asdict(s) for s in scenarios], summary=summary, results=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--duration', type=float, default=10)
    parser.add_argument('--output', type=Path, default=DEFAULT_MODEL.with_suffix('.evaluation.json'))
    args = parser.parse_args()
    result = evaluate(args.model, args.duration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary'], indent=2))
    print(f'Details: {args.output}')


if __name__ == '__main__':
    main()
