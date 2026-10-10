"""Compare LQR, the existing supervised NN and PPO with identical scenarios."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import mujoco
import numpy as np

from control_experiments import Scenario, prepare
from control_timing import CONTROL_DT
from nn_model import DEFAULT_MODEL, NNPredictor
from rl_control import SCENE, RobotRL, body_speed, body_tilt, invalid_physics
from rl_model import RLPredictor
from robot_lqr import RobotLqr, WHEEL_RADIUS
from robot_nn import RobotNN


def evaluate(rl_model, nn_model=DEFAULT_MODEL, duration=8., controllers=('LQR', 'Neural Network', 'Reinforcement Learning')):
    if not np.isfinite(duration) or duration < CONTROL_DT:
        raise ValueError('duration must be at least one control tick')
    rl = RLPredictor(rl_model)
    nn = NNPredictor(nn_model) if 'Neural Network' in controllers else None
    scenarios = [Scenario('balance-positive', .06, duration=duration),
                 Scenario('balance-negative', -.06, heading=.7, duration=duration),
                 Scenario('forward-ramp', .04, targets=(0., .3, .3), duration=duration, ramp=True),
                 Scenario('reverse-ramp', -.04, targets=(0., -.3, -.3), duration=duration, ramp=True),
                 Scenario('forward-stop-reverse', .02, targets=(0., .3, 0., -.3, 0.), duration=duration, ramp=True),
                 Scenario('yaw-mixing', -.03, heading=-.8, targets=(0., .15), yaw=.5, duration=duration, ramp=True)]
    rows = []
    for scenario in scenarios:
        for name in controllers:
            # No shared MjData, controller memory or normalization updates.
            model = mujoco.MjModel.from_xml_path(str(SCENE))
            data = mujoco.MjData(model)
            robot = (RobotLqr(model, data) if name == 'LQR' else RobotNN(model, data, nn)
                     if name == 'Neural Network' else RobotRL(model, data, rl))
            substeps = prepare(model, data, robot, scenario)
            initial_qpos = data.qpos.tolist()
            initial_qvel = data.qvel.tolist()
            errors, wheel_errors, speeds, positions = [], [], [], []
            fell = False
            invalid = False
            for tick in range(round(duration / CONTROL_DT)):
                target = scenario.target_at(tick * CONTROL_DT)
                robot.set_velocity_linear_set_point(target)
                robot.set_yaw(scenario.yaw)
                robot.update_motor_speed()
                mujoco.mj_step(model, data, nstep=substeps)
                invalid = invalid_physics(data)
                if invalid:
                    fell = True
                    break
                errors.append(target - body_speed(data))
                speeds.append(body_speed(data))
                positions.append(data.qpos[:2].copy())
                left = float(data.joint('torso_l_wheel').qvel[0])
                right = float(data.joint('torso_r_wheel').qvel[0])
                wheel_errors.append(target - (-left + right) * WHEEL_RADIUS / 2)
                if body_tilt(data) > rl.config.fall_tilt:
                    fell = True
                    break
            # Inspect the final half separately: a small whole-episode error can
            # hide persistent oscillation, drift, or an early fall.
            settled = None
            if scenario.targets == (0.0,) and not fell:
                tail = slice(len(speeds) // 2, None)
                xy = np.asarray(positions)[tail]
                settled = dict(
                    speed_rms_m_s=float(np.sqrt(np.mean(np.square(np.asarray(speeds)[tail])))),
                    speed_peak_m_s=float(np.max(np.abs(np.asarray(speeds)[tail]))),
                    travel_m=float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))),
                    displacement_m=float(np.linalg.norm(xy[-1] - xy[0])))
                settled['passed'] = bool(settled['speed_rms_m_s'] < .02 and
                                        settled['speed_peak_m_s'] < .05 and
                                        settled['displacement_m'] < .05)
            rows.append(dict(scenario=scenario.name, controller=name, fell=fell, invalid=invalid,
                             stopping=settled,
                             balancing_time_s=(tick + 1) * CONTROL_DT,
                             velocity_tracking_rmse_m_s=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                             wheel_tracking_rmse_m_s=float(np.sqrt(np.mean(np.square(wheel_errors)))) if wheel_errors else None,
                             samples=len(errors), initial_qpos=initial_qpos, initial_qvel=initial_qvel))
    summary = {}
    for name in controllers:
        selected = [r for r in rows if r['controller'] == name]
        summary[name] = dict(falls=sum(r['fell'] for r in selected), episodes=len(selected),
                             fall_rate=float(np.mean([r['fell'] for r in selected])),
                             mean_balancing_time_s=float(np.mean([r['balancing_time_s'] for r in selected])))
        stopping_rows = [r for r in selected if r['scenario'].startswith('balance-')]
        summary[name]['stopping_passes'] = sum(bool(r['stopping'] and r['stopping']['passed'])
                                              for r in stopping_rows)
        summary[name]['stopping_episodes'] = len(stopping_rows)
        for key in ('velocity_tracking_rmse_m_s', 'wheel_tracking_rmse_m_s'):
            values = [r[key] for r in selected if r[key] is not None]
            summary[name][key] = float(np.mean(values)) if values else None
    return dict(rl_model=str(rl_model), nn_model=str(nn_model),
                training_steps=rl.metadata['training']['steps'], duration_s=duration,
                control_dt=CONTROL_DT, physics_dt=float(model.opt.timestep),
                fall_threshold_rad=rl.config.fall_tilt, scenarios=[asdict(s) for s in scenarios],
                definitions={'fall': 'body up-axis tilt from world Z exceeds threshold, or invalid physics',
                             'velocity': 'post-step actual body forward speed (-body Y), RMSE until fall',
                             'wheel_velocity': 'post-step (-left+right)*radius/2, RMSE until fall',
                             'summary': 'unweighted scenario means; read error together with survival/fall rate',
                             'yaw': 'manual mixing only; turning is not a learned objective'},
                summary=summary, results=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rl-model', required=True, type=Path)
    parser.add_argument('--nn-model', default=DEFAULT_MODEL, type=Path)
    parser.add_argument('--duration', default=8., type=float)
    parser.add_argument('--rl-only', action='store_true')
    parser.add_argument('--output', default=Path('models/rl-evaluation.json'), type=Path)
    args = parser.parse_args()
    kwargs = dict(controllers=('Reinforcement Learning',)) if args.rl_only else {}
    result = evaluate(args.rl_model, args.nn_model, args.duration, **kwargs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result['summary'], indent=2, allow_nan=False))
    print(f'Details: {args.output}')


if __name__ == '__main__':
    main()
