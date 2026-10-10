"""Headless PPO training; stdout JSON status is also consumed by Qt signals."""
import argparse
from collections import deque
from dataclasses import asdict
import json
from pathlib import Path
import signal
import sys
import threading
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from rl_control import RLConfig
from rl_env import BalanceEnv
from rl_model import DEFAULT_RL_MODEL, load_training, read_bundle, save_bundle
from rl_validation import validate_policy


class TrainingProgress(BaseCallback):
    def __init__(self, stop, previous=None, emit=None):
        super().__init__()
        previous = previous or {}
        self.stop = stop
        self.emit = emit or (lambda status: print(json.dumps(status, allow_nan=False), flush=True))
        self.episodes = previous.get('episodes', 0)
        self.falls = previous.get('falls', 0)
        self.recent = deque(previous.get('recent', []), maxlen=100)
        self.started = time.monotonic()
        self.last_report = 0.
        self.start_steps = previous.get('steps', 0)

    def status(self, state):
        return dict(state=state, steps=self.model.num_timesteps, episodes=self.episodes, falls=self.falls,
                    mean_reward=float(np.mean([x[0] for x in self.recent])) if self.recent else None,
                    mean_duration=float(np.mean([x[1] for x in self.recent])) if self.recent else None,
                    fps=(self.model.num_timesteps - self.start_steps) / max(.001, time.monotonic() - self.started))

    def _on_step(self):
        for info in self.locals['infos']:
            if 'episode' in info:
                self.episodes += 1
                self.falls += int(info.get('fell', False))
                ep = info['episode']
                self.recent.append([ep['r'], ep['l'] * self.training_env.get_attr('config')[0].control_dt])
        if time.monotonic() - self.last_report >= .5:
            self.emit(self.status('Training'))
            self.last_report = time.monotonic()
        return not self.stop.is_set()


def train(output=DEFAULT_RL_MODEL, steps=100_000, stage=None, resume=None,
          config=None, seed=42, rollout_steps=2048, stop=None, emit=None):
    if steps < 1 or rollout_steps < 64 or rollout_steps % 64:
        raise ValueError('steps > 0; rollout_steps must be a positive multiple of 64')
    torch.set_num_threads(1)
    previous = {}
    if resume:
        _, meta, saved_config, _, _ = read_bundle(resume)
        if config is not None and asdict(config) != asdict(saved_config):
            raise ValueError('Resume restores configuration; use the saved configuration')
        config = saved_config
        stage = stage or meta['training']['stage']
    config = config or RLConfig()
    stage = stage or 'balance'
    env = BalanceEnv(config, stage)
    try:
        if resume:
            model, previous = load_training(resume, env)
        else:
            # Random initialization; no teacher labels, pretrained NN, or LQR actions.
            model = PPO('MlpPolicy', env, device='cpu', seed=seed, n_steps=rollout_steps,
                        batch_size=64, n_epochs=10, learning_rate=3e-4, gamma=.999,
                        gae_lambda=.98,
                        policy_kwargs=dict(net_arch=dict(pi=[32, 32], vf=[32, 32]),
                                           log_std_init=-2.), verbose=0)
        stop = stop if stop is not None else threading.Event()
        callback = TrainingProgress(stop, previous, emit)
        model.learn(total_timesteps=steps, callback=callback, reset_num_timesteps=False)
        state = dict(steps=model.num_timesteps, episodes=callback.episodes, falls=callback.falls,
                     recent=list(callback.recent), stage=stage, seed=previous.get('seed', seed),
                     env_rng=env.np_random.bit_generator.state)
        save_bundle(output, model, config, state)
        validation = None
        if not stop.is_set():
            callback.emit(callback.status('Evaluating'))
            validation = validate_policy(model, config, stage, stop)
            state['validation'] = validation
            save_bundle(output, model, config, state)
        outcome = ('Stopped' if stop.is_set() else
                   'Validated' if validation['passed'] else 'Needs training')
        status = callback.status(outcome)
        status['validation'] = validation
        status['model'] = str(output)
        callback.emit(status)
        return status
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_RL_MODEL)
    parser.add_argument('--steps', type=int, default=100_000, help='additional control steps; rounded up to PPO rollout size')
    parser.add_argument('--stage', choices=('balance', 'velocity'))
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--config', type=Path, help='JSON RLConfig; omitted defaults are used')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rollout-steps', type=int, default=2048)
    parser.add_argument('--stdin-control', action='store_true', help='read stop command from GUI pipe')
    args = parser.parse_args()
    stop = threading.Event()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda *_: stop.set())
    if args.stdin_control:
        def commands():
            for line in sys.stdin:
                if line.strip() == 'stop':
                    stop.set()
            stop.set()  # parent process/pipe closed
        threading.Thread(target=commands, daemon=True).start()
    config = RLConfig(**json.loads(args.config.read_text())) if args.config else None
    train(args.output, args.steps, args.stage, args.resume, config, args.seed, args.rollout_steps, stop)


if __name__ == '__main__':
    main()
