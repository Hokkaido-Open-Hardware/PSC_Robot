"""Real TFLite INT8 inference. Floating point is used only at the I/O boundary."""
import hashlib
import json
from pathlib import Path
import threading

import numpy as np

from nn_model import FEATURES, MODEL_DIR

DEFAULT_INT8_MODEL = MODEL_DIR / 'ROBOT.TFL'


def interpreter_api():
    try:
        from ai_edge_litert.interpreter import Interpreter, OpResolverType
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter, OpResolverType
        except ImportError as exc:
            raise ImportError('Install requirements-int8.txt for NN INT8 control') from exc
    return Interpreter, OpResolverType


def make_interpreter(content):
    Interpreter, resolver = interpreter_api()
    interpreter = Interpreter(model_content=content, num_threads=1,
                              experimental_op_resolver_type=resolver.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
    interpreter.allocate_tensors()
    return interpreter


def quantize(values, scale, zero_point):
    # Explicit ties-to-even rounding, saturation BEFORE casting avoids wraparound.
    return np.clip(np.rint(np.asarray(values) / scale) + zero_point, -128, 127).astype(np.int8)


class INT8Predictor:
    def __init__(self, path=DEFAULT_INT8_MODEL):
        path = Path(path)
        content = path.read_bytes()
        self.metadata = json.loads(path.with_suffix('.json').read_text())
        m = self.metadata
        if (m['version'] != 1 or tuple(m['features']) != FEATURES or
                m['sha256'] != hashlib.sha256(content).hexdigest()):
            raise ValueError('INT8 model/metadata mismatch')
        self.mean = np.asarray(m['mean'], dtype=np.float64)
        self.scale = np.asarray(m['scale'], dtype=np.float64)
        self.output_mean = float(m['output_mean'])
        self.output_scale = float(m['output_scale'])
        if (self.mean.shape != (4,) or self.scale.shape != (4,) or
                not np.isfinite([*self.mean, *self.scale, self.output_mean, self.output_scale]).all()
                or (self.scale <= 0).any() or self.output_scale <= 0):
            raise ValueError('Invalid normalization metadata')
        self.interpreter = make_interpreter(content)
        ins, outs = self.interpreter.get_input_details(), self.interpreter.get_output_details()
        if len(ins) != 1 or len(outs) != 1:
            raise ValueError('Expected one input and one output')
        self.input, self.output = ins[0], outs[0]
        for detail, shape in ((self.input, (1, 4)), (self.output, (1, 1))):
            if detail['dtype'] != np.int8 or tuple(detail['shape']) != shape:
                raise ValueError('Expected fixed [1,4] -> [1,1] INT8 tensors')
            q = detail['quantization_parameters']
            if len(q['scales']) != 1 or not np.isfinite(q['scales'][0]) or q['scales'][0] <= 0:
                raise ValueError('Expected positive per-tensor quantization scale')
        if any(t['dtype'] not in (np.int8, np.int32) for t in self.interpreter.get_tensor_details()):
            raise ValueError('Model is not full integer INT8 (INT32 biases allowed)')
        self.input_scale, self.input_zero = self.input['quantization']
        self.output_qscale, self.output_zero = self.output['quantization']
        self.lock = threading.Lock()

    def encode(self, inputs):
        x = np.asarray(inputs, dtype=np.float64)
        if x.shape != (4,) or not np.isfinite(x).all():
            raise ValueError('Expected four finite robot inputs')
        return quantize((x - self.mean) / self.scale, self.input_scale, self.input_zero).reshape(1, 4)

    def predict(self, inputs):
        encoded = self.encode(inputs)
        with self.lock:
            self.interpreter.set_tensor(self.input['index'], encoded)
            self.interpreter.invoke()
            output = int(self.interpreter.get_tensor(self.output['index'])[0, 0])
        return ((output - self.output_zero) * self.output_qscale * self.output_scale
                + self.output_mean)
