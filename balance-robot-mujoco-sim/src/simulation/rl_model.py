"""Atomic, self-contained RL bundles, separate from legacy NN checkpoints."""
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import random
import tempfile
import zipfile

import numpy as np

from rl_control import RLConfig, contract

DEFAULT_RL_MODEL = Path(__file__).resolve().parents[2] / 'models' / 'robot_rl.rl.zip'


def atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_bundle(path, model, config, training):
    import stable_baselines3
    import gymnasium
    import mujoco
    import torch
    policy = io.BytesIO()
    model.save(policy)  # Includes policy, value network, Adam state and step count.
    rng = io.BytesIO()
    numpy_rng = np.random.get_state()
    torch.save(dict(torch=torch.get_rng_state(), python=random.getstate(),
                    numpy=(numpy_rng[0], numpy_rng[1].tolist(), *numpy_rng[2:])), rng)
    metadata = dict(contract=contract(config), training=training,
                    versions=dict(sb3=stable_baselines3.__version__, gymnasium=gymnasium.__version__,
                                  mujoco=mujoco.__version__, numpy=np.__version__, torch=torch.__version__))
    result = io.BytesIO()
    with zipfile.ZipFile(result, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr('ppo.zip', policy.getvalue())
        bundle.writestr('metadata.json', json.dumps(metadata, indent=2, allow_nan=False))
        bundle.writestr('rng.pt', rng.getvalue())
    atomic_write(path, result.getvalue())


def read_bundle(path):
    payload = Path(path).read_bytes()
    with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
        metadata = json.loads(bundle.read('metadata.json'))
        version = metadata['contract']['version']
        settings = dict(metadata['contract']['config'])
        if version == 1:
            # Preserve old reward semantics when continuing an existing policy.
            settings['reward'] = dict(settings['reward'], tilt_scale=settings['fall_tilt'],
                                      rate=0.0, rate_scale=1.0)
        config = RLConfig(**settings)
        if version not in (1, 2) or metadata['contract'] != contract(config, version):
            raise ValueError('RL observation/action/physics contract is incompatible with this simulator')
        return payload, metadata, config, bundle.read('ppo.zip'), bundle.read('rng.pt')


def load_training(path, env):
    from stable_baselines3 import PPO
    import torch
    _, metadata, config, policy, rng_bytes = read_bundle(path)
    if asdict(config) != asdict(env.config):
        raise ValueError('Cannot resume with a changed RL contract')
    model = PPO.load(io.BytesIO(policy), env=env, device='cpu')
    rng = torch.load(io.BytesIO(rng_bytes), weights_only=True)
    torch.set_rng_state(rng['torch'])
    random.setstate(rng['python'])
    state = rng['numpy']
    np.random.set_state((state[0], np.asarray(state[1], dtype=np.uint32), *state[2:]))
    # Continue from a fresh episode, using the saved environment RNG stream.
    model.set_env(env, force_reset=True)
    env.np_random.bit_generator.state = metadata['training']['env_rng']
    return model, metadata['training']


class RLPredictor:
    def __init__(self, path):
        from stable_baselines3 import PPO
        import torch
        torch.set_num_threads(1)
        self.payload, self.metadata, self.config, policy, _ = read_bundle(path)
        self.model = PPO.load(io.BytesIO(policy), device='cpu')
        self.model.policy.set_training_mode(False)
        if self.model.observation_space.shape != (len(self.config.observation_scale),) or self.model.action_space.shape != (1,):
            raise ValueError('Policy dimensions do not match the RL contract')
        if not all(torch.isfinite(p).all().item() for p in self.model.policy.parameters()):
            raise ValueError('Policy contains non-finite parameters')
        if not np.isfinite(self.predict(np.zeros(len(self.config.observation_scale), dtype=np.float32))).all():
            raise ValueError('Policy produced a non-finite action')
        self.path = Path(path)

    def predict(self, normalized_observation):
        action, _ = self.model.predict(normalized_observation, deterministic=True)
        return action

    def save(self, path):
        # Copy the exact immutable snapshot, never a concurrently training model.
        atomic_write(path, self.payload)


def validation_message(validation):
    if not validation or not validation.get('completed'):
        return 'Not evaluated; step count does not prove balance. Continue training to evaluate.'
    result = 'Passed' if validation['passed'] else 'Needs training'
    return (f"{result}: {validation['passes']}/{validation['episodes']} trials passed; "
            f"{validation['falls']} falls; mean duration {validation['mean_duration_s']:.2f} / "
            f"{validation['episode_seconds']:g} s.")
