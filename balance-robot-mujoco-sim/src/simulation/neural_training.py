"""Collect LQR demonstrations and train/export a 4 -> 16 -> 16 -> 1 MLP."""
import argparse
from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from control_experiments import (CONTROL_DT, FALL_PITCH, SCENE, applied_velocity,
                                 Scenario, control_input, prepare)
from nn_model import DEFAULT_MODEL, FEATURES, atomic_npz, build_mlp, export_model
from robot_lqr import RobotLqr
from auto_commands import (AutoCommands, SCENARIOS, MIN_EPISODES, MIN_DURATION,
                           DEFAULT_DURATION, AUTO_SPEED_LIMIT, AUTO_YAW_LIMIT,
                           SPEED_SLEW, YAW_SLEW, MIN_HOLD, MAX_HOLD)


@dataclass(frozen=True)
class TrainingStatus:
    state: str = 'Idle'
    progress: float = 0
    samples: int = 0
    epoch: int = 0
    total_epochs: int = 0
    training_loss: float | None = None
    validation_loss: float | None = None
    best_validation_loss: float | None = None
    learning_rate: float | None = None
    elapsed: float = 0
    message: str = ''
    current_episode: int = 0
    total_episodes: int = 0
    auto_speed: float = 0
    auto_yaw: float = 0
    scenario: str = ''
    episode_elapsed: float = 0
    falls: int = 0


def collect_data(episodes, duration, seed, publish):
    if episodes < MIN_EPISODES or not np.isfinite(duration) or duration < MIN_DURATION:
        raise ValueError(f'Need >={MIN_EPISODES} episodes and duration >={MIN_DURATION}s')
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    rng = np.random.default_rng(seed)
    xs, ys, groups, patterns, commands, targets, times = [], [], [], [], [], [], []
    records = []
    falls = 0
    for episode in range(episodes):
        name = SCENARIOS[episode % len(SCENARIOS)]
        scenario = Scenario(name, float(rng.uniform(-.08, .08)),
                            heading=float(rng.uniform(-np.pi, np.pi)),
                            roll=float(rng.uniform(-.03, .03)), duration=duration)
        auto = AutoCommands(name, duration, rng)
        # Fresh physics and controller state (including filters) every episode.
        data = mujoco.MjData(model)
        robot = RobotLqr(model, data)
        substeps = prepare(model, data, robot, scenario)
        count = round(duration / CONTROL_DT)
        start = len(xs)
        fell = False
        publish(current_episode=episode + 1, total_episodes=episodes,
                scenario=name, episode_elapsed=0, auto_speed=0, auto_yaw=0,
                falls=falls, progress=episode / episodes, samples=len(xs))
        for tick in range(count):
            elapsed = tick * CONTROL_DT
            speed, yaw = auto.step(elapsed, CONTROL_DT)
            robot.set_velocity_linear_set_point(speed)
            robot.set_yaw(yaw)
            robot.update_motor_speed()
            inputs = control_input(robot)
            target = applied_velocity(robot)
            if not np.isfinite((*inputs, target)).all():
                raise ValueError('Non-finite teacher sample')
            xs.append(inputs)
            ys.append(target)
            groups.append(episode)
            patterns.append(name)
            commands.append((speed, yaw))
            targets.append(auto.target)
            times.append(elapsed)
            mujoco.mj_step(model, data, nstep=substeps)
            if data.warning.number.any() or not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError('Invalid MuJoCo physics during collection')
            # Angle between body up and world up catches sideways falls too,
            # independently of heading or Euler angle wrapping.
            fell = data.body('robot_body').xmat[8] < np.cos(FALL_PITCH)
            if fell:
                falls += 1
            if tick % 20 == 0 or fell or tick == count - 1:
                publish(progress=(episode + (tick + 1) / count) / episodes,
                        samples=len(xs), auto_speed=speed, auto_yaw=yaw,
                        episode_elapsed=(tick + 1) * CONTROL_DT, falls=falls)
            if fell:
                break
        records.append(dict(episode=episode, scenario=name, initial_pitch=scenario.pitch,
                            initial_roll=scenario.roll, heading=scenario.heading,
                            elapsed=(tick + 1) * CONTROL_DT, fell=bool(fell),
                            samples=len(xs) - start,
                            segments=[asdict(segment) for segment in auto.segments]))
        publish(progress=(episode + 1) / episodes, samples=len(xs))
    details = dict(scenario=np.asarray(patterns), commands=np.asarray(commands, np.float32),
                   target_commands=np.asarray(targets, np.float32),
                   episode_time=np.asarray(times, np.float32), records=records, falls=falls)
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32)[:, None], np.asarray(groups), details


def balanced_split(groups, patterns, rng):
    """Hold out whole episodes per pattern, then equalize each split's samples.

    Sampling across the full trajectories retains late reversals/turns even if
    another scenario falls early. No duplicates and no adjacent-tick leakage.
    """
    val_groups = []
    for name in SCENARIOS:
        episodes = np.unique(groups[patterns == name])
        if len(episodes) < 2:
            raise ValueError(f'Need at least two episodes for {name}')
        val_groups.extend(rng.permutation(episodes)[:max(1, round(len(episodes) * .2))])
    validation = np.isin(groups, val_groups)
    selected = []
    for split in (False, True):
        buckets = [np.flatnonzero((patterns == name) & (validation == split)) for name in SCENARIOS]
        quota = min(map(len, buckets))
        if quota < 2:
            raise ValueError('Too few samples for balanced training/validation')
        selected.extend(rng.choice(bucket, quota, replace=False) for bucket in buckets)
    indices = np.sort(np.concatenate(selected))
    return indices, validation[indices], np.asarray(val_groups, dtype=int)


def train(output=DEFAULT_MODEL, episodes=120, duration=DEFAULT_DURATION, epochs=240, seed=42,
          learning_rate=.003, callback=lambda status: None):
    if (episodes < MIN_EPISODES or not np.isfinite(duration) or duration < MIN_DURATION
            or epochs < 1 or not np.isfinite(learning_rate) or learning_rate <= 0):
        raise ValueError(f'Need >={MIN_EPISODES} episodes, duration >={MIN_DURATION}s, positive epochs and learning rate')
    import torch
    # This runs in a dedicated process. Leave CPU capacity for the GUI/simulator.
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    output = Path(output)
    started = time.monotonic()
    status = TrainingStatus(total_epochs=epochs, total_episodes=episodes, learning_rate=learning_rate)

    def publish(**changes):
        nonlocal status
        status = replace(status, elapsed=time.monotonic() - started, **changes)
        callback(status)

    publish(state='Collecting Data')
    x, y, groups, details = collect_data(episodes, duration, seed, publish)
    collected_samples = len(x)
    indices, validation, val_groups = balanced_split(groups, details['scenario'], rng)
    x, y, groups = x[indices], y[indices], groups[indices]
    sample_metadata = {key: details[key][indices] for key in
                       ('scenario', 'commands', 'target_commands', 'episode_time')}
    counts = {name: dict(collected=int(np.sum(details['scenario'] == name)),
                        training=int(np.sum((sample_metadata['scenario'] == name) & ~validation)),
                        validation=int(np.sum((sample_metadata['scenario'] == name) & validation)))
              for name in SCENARIOS}
    mean = x[~validation].mean(axis=0)
    scale = np.maximum(x[~validation].std(axis=0), 1e-6)
    output_mean = float(y[~validation].mean())
    output_scale = max(float(y[~validation].std()), 1e-6)
    atomic_npz(output.with_suffix('.data.npz'), inputs=x, targets=y, episode=groups,
               validation=validation, features=np.array(FEATURES), seed=np.array(seed),
               **sample_metadata)
    xt = torch.from_numpy((x[~validation] - mean) / scale)
    yt = torch.from_numpy((y[~validation] - output_mean) / output_scale)
    xv = torch.from_numpy((x[validation] - mean) / scale)
    yv = torch.from_numpy((y[validation] - output_mean) / output_scale)
    model = build_mlp()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Final full-batch LBFGS epochs refine this very small regression network.
    adam_epochs = max(1, round(epochs * .75))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=adam_epochs,
                                                           eta_min=learning_rate * .03)
    criterion = torch.nn.MSELoss()
    best = float('inf')
    best_state = None
    history = []
    publish(state='Training', progress=1,
            message=f'{len(x)} balanced samples retained from {collected_samples} collected')
    for epoch in range(1, epochs + 1):
        model.train()
        if epoch <= adam_epochs:
            order = torch.randperm(len(xt))
            for indices in order.split(1024):
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xt[indices]), yt[indices])
                loss.backward()
                optimizer.step()
            lr = optimizer.param_groups[0]['lr']
            scheduler.step()
        else:
            if epoch == adam_epochs + 1:
                optimizer = torch.optim.LBFGS(model.parameters(), lr=.8, max_iter=10,
                                              tolerance_grad=1e-10, tolerance_change=1e-12,
                                              line_search_fn='strong_wolfe')
            def closure():
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(xt), yt)
                loss.backward()
                return loss
            optimizer.step(closure)
            lr = optimizer.param_groups[0]['lr']
        model.eval()
        with torch.inference_mode():
            # Display/report physical MSE in (rad/s)^2, not normalized loss.
            training_loss = float(criterion(model(xt), yt)) * output_scale ** 2
            validation_loss = float(criterion(model(xv), yv)) * output_scale ** 2
        if not np.isfinite([training_loss, validation_loss]).all():
            raise ValueError('Training loss is not finite')
        if validation_loss < best:
            best = validation_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        publish(epoch=epoch, training_loss=training_loss, validation_loss=validation_loss,
                best_validation_loss=best, learning_rate=lr)
        history.append(asdict(status))
    model.load_state_dict(best_state)
    model.eval()
    publish(state='Validation')
    with torch.inference_mode():
        final_validation = float(criterion(model(xv), yv)) * output_scale ** 2
    if not np.isfinite(final_validation):
        raise ValueError('Final validation loss is not finite')
    publish(validation_loss=final_validation)
    publish(state='Saving Model')
    checkpoint = dict(version=1, architecture=[4, 16, 16, 1], activation='LeakyReLU(0.1)',
                      features=list(FEATURES), state_dict=best_state,
                      mean=torch.tensor(mean), scale=torch.tensor(scale),
                      output_mean=output_mean, output_scale=output_scale,
                      seed=seed, best_validation_loss=best)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.pt.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(output.with_suffix('.pt'))
    export_model(output, model, mean, scale, output_mean, output_scale)
    report = dict(seed=seed, episodes=episodes, duration=duration, epochs=epochs,
                  versions=dict(torch=torch.__version__, mujoco=mujoco.__version__, numpy=np.__version__),
                  collected_samples=collected_samples, falls=details['falls'],
                  scenario_counts=counts, collection_episodes=details['records'],
                  auto_commands=dict(speed_limit=AUTO_SPEED_LIMIT, yaw_limit=AUTO_YAW_LIMIT,
                                     speed_slew=SPEED_SLEW, yaw_slew=YAW_SLEW,
                                     hold_seconds=[MIN_HOLD, MAX_HOLD]),
                  samples=len(x), training_samples=int((~validation).sum()),
                  validation_samples=int(validation.sum()), validation_episodes=val_groups.tolist(),
                  features=list(FEATURES), input_min=x.min(0).tolist(), input_max=x.max(0).tolist(),
                  target_min=float(y.min()), target_max=float(y.max()),
                  best_validation_loss=best, validation_rmse=best ** .5,
                  loss_units='(rad/s)^2', elapsed=time.monotonic() - started, history=history)
    output.with_suffix('.training.json').write_text(json.dumps(report, indent=2) + '\n')
    publish(state='Completed', message=str(output))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--episodes', type=int, default=120)
    parser.add_argument('--duration', type=float, default=DEFAULT_DURATION)
    parser.add_argument('--epochs', type=int, default=240)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--learning-rate', type=float, default=.003)
    parser.add_argument('--json-status', action='store_true')
    args = parser.parse_args()
    def report(status):
        if args.json_status:
            print(json.dumps(asdict(status)), flush=True)
        elif status.state != 'Collecting Data' or status.progress == 1:
            print(f'{status.state}: samples={status.samples}, epoch={status.epoch}/{status.total_epochs}, '
                  f'val MSE={status.validation_loss}, elapsed={status.elapsed:.1f}s', flush=True)
    try:
        train(args.output, args.episodes, args.duration, args.epochs, args.seed,
              args.learning_rate, report)
    except Exception as exc:
        if args.json_status:
            print(json.dumps(asdict(TrainingStatus(state='Failed', message=str(exc)))), flush=True)
        raise


if __name__ == '__main__':
    main()
