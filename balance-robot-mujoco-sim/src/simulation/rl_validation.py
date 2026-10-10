"""Deterministic held-out rollouts, independent of PPO's exploration rewards."""
from dataclasses import replace

import numpy as np

from rl_control import body_speed
from rl_env import BalanceEnv


def validate_policy(model, config, stage, stop=None):
    # Short training episodes must not yield a spurious validation pass. Leave
    # enough time to ramp to either speed limit and back, then hold zero.
    # A 12-second training rollout can hide oscillations that grow into a fall
    # just after its time limit. Validate over a longer uninterrupted run.
    duration = max(30., config.episode_seconds)
    if stage == 'velocity':
        duration = max(duration, 2 * config.target_limit / config.target_slew +
                       3 * config.target_hold_seconds + config.control_dt)
    config = replace(config, episode_seconds=duration)
    # Always check stopping, including after velocity training. Passing only
    # moving targets can hide a policy that drifts or has forgotten balance.
    stages = ('balance',) if stage == 'balance' else ('balance', 'velocity')
    rows = []
    for test_stage in stages:
        env = BalanceEnv(config, test_stage)
        try:
            for seed in (10001, 10002, 10003, 10004):
                obs, _ = env.reset(seed=seed)
                errors, speeds = [], []
                for tick in range(env.max_ticks):
                    if stop is not None and stop.is_set():
                        return dict(completed=False, passed=False, results=rows)
                    action, _ = model.predict(obs, deterministic=True)
                    obs, _, terminated, truncated, info = env.step(action)
                    if not info['invalid']:
                        errors.append(info['velocity_error'])
                        speeds.append(body_speed(env.data))
                    if terminated or truncated:
                        break
                survived = not terminated and truncated
                rmse = float(np.sqrt(np.mean(np.square(errors)))) if errors else None
                tail = speeds[len(speeds) // 2:]
                stopped_rms = float(np.sqrt(np.mean(np.square(tail)))) if tail else None
                stopped_peak = float(np.max(np.abs(tail))) if tail else None
                passed = survived and (stopped_rms < .02 and stopped_peak < .05
                                       if test_stage == 'balance' else rmse < .08)
                rows.append(dict(stage=test_stage, seed=seed, fell=info['fell'],
                                 invalid=info['invalid'], duration_s=(tick + 1) * config.control_dt,
                                 tracking_rmse_m_s=rmse, stopping_rms_m_s=stopped_rms,
                                 stopping_peak_m_s=stopped_peak, passed=bool(passed)))
        finally:
            env.close()
    return dict(completed=True, passed=all(row['passed'] for row in rows),
                episodes=len(rows), falls=sum(row['fell'] for row in rows),
                passes=sum(row['passed'] for row in rows),
                mean_duration_s=float(np.mean([row['duration_s'] for row in rows])),
                episode_seconds=config.episode_seconds,
                criteria=dict(balance_tail_rms_m_s=.02, balance_tail_peak_m_s=.05,
                              velocity_rmse_m_s=.08, require_no_falls=True),
                results=rows)
