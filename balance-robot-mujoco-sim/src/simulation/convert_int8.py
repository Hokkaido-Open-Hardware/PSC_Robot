"""Convert frozen existing weights with actual MuJoCo representative samples.

Run in requirements-int8-convert.txt environment; never trains or overwrites the
source. Normalization remains at the I/O boundary exactly as in NNPredictor.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from nn_model import DEFAULT_MODEL, FEATURES, NNPredictor


def layers(parameters, relu_only=False):
    p = parameters
    w = [p[f'w{i}'].astype(np.float32) for i in range(3)]
    b = [p[f'b{i}'].astype(np.float32) for i in range(3)]
    if relu_only:
        # LReLU(z) = ReLU(z) - alpha * ReLU(-z). Carry both signs
        # into the next dense layer; this is an exact graph rewrite, not training.
        alpha = float(p['slope'])
        w = [np.concatenate([w[0], -w[0]]),
             np.block([[w[1], -alpha * w[1]], [-w[1], alpha * w[1]]]),
             np.concatenate([w[2], -alpha * w[2]], axis=1)]
        b = [np.concatenate([b[0], -b[0]]), np.concatenate([b[1], -b[1]]), b[2]]
    return w, b


def inspect(content):
    from tensorflow.lite.python import schema_py_generated as schema
    model = schema.Model.GetRootAsModel(content, 0)
    graph = model.Subgraphs(0)
    names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    types = {v: k for k, v in vars(schema.TensorType).items() if isinstance(v, int)}
    tensors, ops = [], []
    for i in range(graph.TensorsLength()):
        t = graph.Tensors(i)
        q = t.Quantization()
        tensors.append(dict(index=i, name=t.Name().decode(), dtype=types[t.Type()],
                            shape=t.ShapeAsNumpy().tolist(),
                            scales=q.ScaleAsNumpy().tolist() if q and q.ScaleLength() else [],
                            zero_points=q.ZeroPointAsNumpy().tolist() if q and q.ZeroPointLength() else []))
    for i in range(graph.OperatorsLength()):
        op = graph.Operators(i)
        code = model.OperatorCodes(op.OpcodeIndex())
        row = dict(name=names[code.BuiltinCode()], version=code.Version(),
                   inputs=op.InputsAsNumpy().tolist(), outputs=op.OutputsAsNumpy().tolist())
        if row['name'] == 'FULLY_CONNECTED':
            table = op.BuiltinOptions()
            fc = schema.FullyConnectedOptions()
            fc.Init(table.Bytes, table.Pos)
            row['fused_activation'] = {0: 'NONE', 1: 'RELU'}.get(fc.FusedActivationFunction(), 'OTHER')
        ops.append(row)
    return dict(tensors=tensors, operators=ops, inputs=graph.InputsAsNumpy().tolist(),
                outputs=graph.OutputsAsNumpy().tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--relu-only', action='store_true')
    parser.add_argument('--samples', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-pitch', type=float, default=None,
                        help='Optional calibration envelope in rad; actual samples only, no clipping')
    args = parser.parse_args()
    if args.output.resolve() == args.model.resolve() or args.samples < 8:
        parser.error('Use a new output file and >=8 representative samples')
    import tensorflow as tf
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    predictor = NNPredictor(args.model)
    p = predictor.arrays
    with np.load(args.calibration, allow_pickle=False) as archive:
        states = archive['inputs']
        calibration_seed = int(archive['seed'])
    if args.max_pitch is not None:
        states = states[np.abs(states[:, 0]) <= args.max_pitch]
    if states.ndim != 2 or states.shape[1] != 4 or not len(states) or not np.isfinite(states).all():
        raise ValueError('Invalid representative states')
    normalized = ((states - p['mean']) / p['scale']).astype(np.float32)
    rng = np.random.default_rng(args.seed)
    indices = np.unique(np.concatenate([
        rng.choice(len(states), min(args.samples, len(states)), replace=False),
        normalized.argmin(axis=0), normalized.argmax(axis=0)]))
    representative = normalized[indices]
    w, b = layers(p, args.relu_only)
    class Network(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec([1, 4], tf.float32, name='normalized_state')])
        def infer(self, x):
            for i in range(3):
                x = tf.matmul(x, tf.constant(w[i].T)) + tf.constant(b[i])
                if i < 2:
                    x = tf.nn.relu(x) if args.relu_only else tf.nn.leaky_relu(x, alpha=float(p['slope']))
            return x
    network = Network()
    concrete = network.infer.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], network)
    float_content = converter.convert()
    # Confirm graph rewrite and conversion BEFORE quantization against source.
    interpreter = tf.lite.Interpreter(model_content=float_content, num_threads=1)
    interpreter.allocate_tensors()
    inp, out = interpreter.get_input_details()[0], interpreter.get_output_details()[0]
    comparison = []
    for index in indices[:1024]:
        interpreter.set_tensor(inp['index'], normalized[index:index + 1])
        interpreter.invoke()
        actual = interpreter.get_tensor(out['index'])[0, 0] * p['output_scale'] + p['output_mean']
        comparison.append(float(actual - predictor.predict(states[index])))
    if max(abs(e) for e in comparison) > .001:
        raise ValueError('Float TFLite does not match original weights')
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: ([x.reshape(1, 4)] for x in representative)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    # PSC-ONE currently requires per-tensor weights, FC v4.
    converter._experimental_disable_per_channel = True
    content = converter.convert()
    graph = inspect(content)
    if any(t['dtype'] not in ('INT8', 'INT32') for t in graph['tensors']):
        raise ValueError('Conversion produced non-integer tensors')
    for i in graph['inputs'] + graph['outputs']:
        if graph['tensors'][i]['dtype'] != 'INT8':
            raise ValueError('I/O must be INT8')
    metadata = dict(version=1, features=FEATURES, sha256=hashlib.sha256(content).hexdigest(),
                    source_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                    mean=p['mean'].tolist(), scale=p['scale'].tolist(),
                    output_mean=float(p['output_mean']), output_scale=float(p['output_scale']),
                    architecture=[4] + [len(v) for v in b], relu_only=args.relu_only,
                    macs=sum(v.size for v in w), parameters=sum(v.size for v in w + b),
                    file_size_bytes=len(content), float_tflite_size_bytes=len(float_content),
                    float_tflite_max_error_rad_s=max(abs(e) for e in comparison),
                    tensorflow_version=tf.__version__, calibration=dict(
                        sha256=hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
                        seed=calibration_seed, selection_seed=args.seed, samples=len(representative),
                        max_pitch=args.max_pitch, normalized_min=representative.min(axis=0).tolist(),
                        normalized_max=representative.max(axis=0).tolist()), **graph)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(content)
    args.output.with_suffix('.fp32.tflite').write_bytes(float_content)
    args.output.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps({k: metadata[k] for k in ('architecture', 'macs', 'file_size_bytes', 'operators')}, indent=2))


if __name__ == '__main__':
    main()
