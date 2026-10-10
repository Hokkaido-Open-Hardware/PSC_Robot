"""PyTorch training model and a small, equivalent NumPy inference export.

PyTorch is imported only for training, so the LQR GUI needs no torch installation.
"""
from pathlib import Path
import os
import tempfile

import numpy as np

FEATURES = ('pitch', 'filtered_pitch_rate', 'filtered_wheel_angular_velocity',
            'target_linear_velocity')
MODEL_DIR = Path(__file__).resolve().parents[2] / 'models'
DEFAULT_MODEL = MODEL_DIR / 'robot_nn.npz'
NEGATIVE_SLOPE = 0.1


def build_mlp():
    from torch import nn
    return nn.Sequential(
        nn.Linear(4, 16), nn.LeakyReLU(NEGATIVE_SLOPE),
        nn.Linear(16, 16), nn.LeakyReLU(NEGATIVE_SLOPE), nn.Linear(16, 1),
    )


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix='.npz')
    try:
        with os.fdopen(fd, 'wb') as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def export_model(path, model, mean, scale, output_mean, output_scale):
    arrays = dict(version=np.array(1), features=np.array(FEATURES),
                  mean=mean, scale=scale, output_mean=np.array(output_mean),
                  output_scale=np.array(output_scale), slope=np.array(NEGATIVE_SLOPE))
    for i, layer in enumerate((model[0], model[2], model[4])):
        arrays[f'w{i}'] = layer.weight.detach().cpu().numpy()
        arrays[f'b{i}'] = layer.bias.detach().cpu().numpy()
    atomic_npz(path, **arrays)


class NNPredictor:
    """Read-only inference weights; no training or file access in control ticks."""
    def __init__(self, path=DEFAULT_MODEL):
        with np.load(path, allow_pickle=False) as archive:
            self.arrays = {key: archive[key].astype(np.float64)
                           for key in ('mean', 'scale', 'output_mean', 'output_scale',
                                       'slope', 'w0', 'b0', 'w1', 'b1', 'w2', 'b2')}
            if archive['version'].item() != 1 or tuple(archive['features']) != FEATURES:
                raise ValueError('Unsupported NN model format or input order')
        shapes = dict(mean=(4,), scale=(4,), output_mean=(), output_scale=(), slope=(),
                      w0=(16, 4), b0=(16,), w1=(16, 16), b1=(16,), w2=(1, 16), b2=(1,))
        for key, shape in shapes.items():
            value = self.arrays[key]
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f'Invalid model parameter: {key}')
            value.flags.writeable = False
        if (self.arrays['scale'] <= 0).any() or self.arrays['output_scale'] <= 0:
            raise ValueError('Normalization scales must be positive')
        if self.arrays['slope'].item() != NEGATIVE_SLOPE:
            raise ValueError('Unsupported activation')

    def predict(self, inputs):
        p = self.arrays
        x = (np.asarray(inputs, dtype=np.float64) - p['mean']) / p['scale']
        for i in range(2):
            x = x @ p[f'w{i}'].T + p[f'b{i}']
            x = np.maximum(x, NEGATIVE_SLOPE * x)
        return ((x @ p['w2'].T + p['b2']) * p['output_scale'] + p['output_mean'])[..., 0]
