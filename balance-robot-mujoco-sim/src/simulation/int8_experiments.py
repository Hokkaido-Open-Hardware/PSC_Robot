"""Collect independent MuJoCo calibration states and compare real controllers."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import time

import mujoco
import numpy as np

from control_experiments import (CONTROL_DT, FALL_PITCH, SCENE, Scenario,
                                 control_input, lqr_reference, prepare)
from int8_model import DEFAULT_INT8_MODEL, INT8Predictor
from nn_model import DEFAULT_MODEL, NNPredictor, atomic_npz
from robot_lqr import MAX_MOTOR_VEL, RobotLqr
from robot_nn import RobotNN


def scenarios(seed, duration, repeats):
    rng = np.random.default_rng(seed)
    result = []
    for category in ('balance', 'disturbance', 'drive', 'stress'):
        for i in range(repeats):
            sign = (-1) ** i
            targets = ((0, 0, sign * .15, sign * .3, 0) if category == 'drive' else
                       (0, sign * .6, -sign * .6, 0) if category == 'stress' else (0,))
            s = Scenario(f'{category}-{i}', sign * float(rng.uniform(.04, .16)),
                         heading=float(rng.uniform(-np.pi, np.pi)),
                         pitch_rate=float(rng.uniform(-.15, .15)),
                         roll=float(rng.uniform(-.02, .02)), targets=targets,
                         duration=duration, ramp=category == 'drive')
            result.append((s, sign * (.015 + .01 * (i % 3)) if category == 'disturbance' else 0.0))
    return result


def rollout(model, scenario, torque, predictor=None):
    data = mujoco.MjData(model)
    robot = RobotLqr(model, data) if predictor is None else RobotNN(model, data, predictor)
    substeps = prepare(model, data, robot, scenario)
    states, pitches, tilts, outputs = [], [], [], []
    body = model.body('robot_body').id
    pulse_start = scenario.duration * .35
    pulse_end = pulse_start + .1
    recovered, stable = None, 0
    fell = False
    for tick in range(round(scenario.duration / CONTROL_DT)):
        t = tick * CONTROL_DT
        robot.set_velocity_linear_set_point(scenario.target_at(t))
        robot.set_yaw(scenario.yaw)
        robot.update_motor_speed()
        states.append(control_input(robot))
        outputs.append(float(robot.control_status.control_output))
        data.xfrc_applied[:] = 0
        if torque and pulse_start <= t < pulse_end:
            # Fixed world axis aligned with the initial robot pitch axis;
            # identical physical pulse, regardless of controller trajectory.
            data.xfrc_applied[body, 3:6] = torque * np.array([
                np.cos(scenario.heading), np.sin(scenario.heading), 0])
        mujoco.mj_step(model, data, nstep=substeps)
        if data.warning.number.any() or not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError(f'Invalid physics: {scenario.name}')
        pitch = abs(robot.get_pitch())
        tilt = np.arccos(np.clip(data.body('robot_body').xmat[8], -1, 1))
        pitches.append(pitch)
        tilts.append(tilt)
        if torque and data.time >= pulse_end:
            stable = stable + 1 if tilt < np.deg2rad(2) else 0
            if stable >= round(.5 / CONTROL_DT) and recovered is None:
                recovered = max(0, float(data.time - .5 - pulse_end))
        if tilt > FALL_PITCH:
            fell = True
            break
    x = np.asarray(states)
    output_error = np.clip(outputs, -MAX_MOTOR_VEL, MAX_MOTOR_VEL) - lqr_reference(x)
    row = dict(scenario=scenario.name, category=scenario.name.rsplit('-', 1)[0],
               fell=bool(fell), balancing_time_s=float(data.time), samples=len(states),
               maximum_pitch_deg=float(np.rad2deg(max(pitches))),
               rms_pitch_deg=float(np.rad2deg(np.sqrt(np.mean(np.square(pitches))))),
               maximum_tilt_deg=float(np.rad2deg(max(tilts))),
               output_vs_lqr_rmse_rad_s=float(np.sqrt(np.mean(output_error ** 2))),
               output_vs_lqr_max_error_rad_s=float(np.max(np.abs(output_error))),
               disturbance_torque_nm=torque, recovery_s=recovered,
               recovered=bool(recovered is not None and not fell) if torque else None)
    return row, x


def errors(values):
    values = np.asarray(values)
    return dict(rmse=float(np.sqrt(np.mean(values ** 2))),
                mae=float(np.mean(np.abs(values))), maximum=float(np.max(np.abs(values))),
                p99=float(np.percentile(np.abs(values), 99)))


def benchmark(fn, states):
    for x in states[:100]:
        fn(x)
    samples = []
    for x in states:
        start = time.perf_counter_ns()
        fn(x)
        samples.append((time.perf_counter_ns() - start) / 1000)
    return dict(samples=len(samples), mean_us=float(np.mean(samples)),
                median_us=float(np.median(samples)), p95_us=float(np.percentile(samples, 95)),
                max_us=float(max(samples)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['collect', 'evaluate'])
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--int8', type=Path, default=DEFAULT_INT8_MODEL)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261009)
    parser.add_argument('--duration', type=float, default=20)
    parser.add_argument('--repeats', type=int, default=6)
    args = parser.parse_args()
    if not np.isfinite(args.duration) or args.duration < 3 or args.repeats < 1:
        parser.error('duration >=3 and repeats >=1 required')
    fp = NNPredictor(args.model)
    suite = scenarios(args.seed, args.duration, args.repeats)
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    predictors = {'LQR': None, 'NN FP32': fp}
    if args.action == 'evaluate':
        predictors['NN INT8'] = INT8Predictor(args.int8)
    rows, states, groups = [], [], []
    for episode, (scenario, torque) in enumerate(suite):
        for name, predictor in predictors.items():
            row, x = rollout(model, scenario, torque, predictor)
            row['controller'] = name
            rows.append(row)
            states.append(x)
            groups.extend([episode] * len(x))
            print(f'{scenario.name} {name}: {row["balancing_time_s"]:.2f}s '
                  f'fall={row["fell"]} rms={row["rms_pitch_deg"]:.3f}deg', flush=True)
    x = np.concatenate(states)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = dict(seed=args.seed, duration_s=args.duration, control_dt_s=CONTROL_DT,
                  physics_dt_s=model.opt.timestep, versions=dict(mujoco=mujoco.__version__,
                  numpy=np.__version__, python=platform.python_version()),
                  source_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                  scenes_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in (SCENE, SCENE.with_name('robot-02.xml'))},
                  scenarios=[dict(**asdict(s), torque_nm=t) for s, t in suite], results=rows)
    if args.action == 'collect':
        atomic_npz(args.output, inputs=x.astype(np.float32), episode=np.asarray(groups), seed=np.array(args.seed))
        result['samples'] = len(x)
        result['input_min'] = x.min(axis=0).tolist()
        result['input_max'] = x.max(axis=0).tolist()
        args.output.with_suffix('.json').write_text(json.dumps(result, indent=2) + '\n')
        return
    q = predictors['NN INT8']
    # Fixed shared held-out states sampled across ALL independent trajectories.
    subset = x[np.linspace(0, len(x) - 1, min(12000, len(x)), dtype=int)]
    a, b = fp.predict(subset), np.asarray([q.predict(s) for s in subset])
    result['inference_error_rad_s'] = errors(b - a)
    result['inference_error_clipped_rad_s'] = errors(np.clip(b, -500, 500) - np.clip(a, -500, 500))
    normalized = (subset - q.mean) / q.scale
    codes = np.rint(normalized / q.input_scale) + q.input_zero
    result['input_saturation_fraction_by_feature'] = np.mean((codes < -128) | (codes > 127), axis=0).tolist()
    output_codes = np.rint(((b - q.output_mean) / q.output_scale) / q.output_qscale) + q.output_zero
    result['output_boundary_fraction'] = float(np.mean((output_codes == -128) | (output_codes == 127)))
    result['heldout_input_min'] = subset.min(axis=0).tolist()
    result['heldout_input_max'] = subset.max(axis=0).tolist()
    atomic_npz(args.output.with_suffix('.states.npz'), inputs=subset.astype(np.float32),
               fp32_output=a, int8_output=b, seed=np.array(args.seed))
    result['int8_sha256'] = q.metadata['sha256']
    result['definitions'] = dict(
        fall='first body tilt >45 degrees (includes sideways falls)',
        recovery='first <2 degree body tilt interval lasting 0.5s after 0.1s torque pulse; null if absent',
        output_error='applied clipped output minus LQR on the SAME state and filter history, rad/s',
        inference_error='INT8 minus original NumPy predictor on shared held-out states, rad/s',
        timing='single-thread desktop wall time, I/O normalization included; excludes physics and sensing',
        fp32='existing FP32 trained weights; original NumPy predictor computes in float64',
        summary='unweighted episode means; maximum is global maximum; stress failures reported separately')
    result['summary'] = {}
    for name in predictors:
        result['summary'][name] = {}
        for category in ('all', 'operating', 'balance', 'disturbance', 'drive', 'stress'):
            selected = [r for r in rows if r['controller'] == name and
                        (category == 'all' or r['category'] == category or
                         (category == 'operating' and r['category'] != 'stress'))]
            entry = {k: float(np.mean([r[k] for r in selected])) for k in
                     ('balancing_time_s', 'rms_pitch_deg', 'output_vs_lqr_rmse_rad_s')}
            entry.update(episodes=len(selected), falls=sum(r['fell'] for r in selected),
                         fall_rate=float(np.mean([r['fell'] for r in selected])),
                         maximum_pitch_deg=max(r['maximum_pitch_deg'] for r in selected),
                         maximum_tilt_deg=max(r['maximum_tilt_deg'] for r in selected))
            disturbed = [r for r in selected if r['disturbance_torque_nm']]
            if disturbed:
                recovered = [r['recovery_s'] for r in disturbed if r['recovered']]
                entry['recovered_episodes'] = len(recovered)
                entry['mean_recovery_s'] = float(np.mean(recovered)) if recovered else None
            result['summary'][name][category] = entry
    result['inference_timing'] = {name: benchmark(fn, subset[:4000]) for name, fn in
                                [('LQR', lqr_reference), ('NN FP32', fp.predict), ('NN INT8', q.predict)]}
    result['baseline_equivalent_survival'] = all(
        next(r for r in rows if r['controller'] == 'NN INT8' and r['scenario'] == baseline['scenario'])['balancing_time_s']
        >= baseline['balancing_time_s'] - CONTROL_DT * 1.1
        for baseline in rows if baseline['controller'] == 'NN FP32' and not baseline['fell'])
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v['all'] for k, v in result['summary'].items()}, indent=2))


if __name__ == '__main__':
    main()
