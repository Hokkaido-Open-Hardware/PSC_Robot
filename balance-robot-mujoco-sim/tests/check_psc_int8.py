"""Read-only PSC-ONE production CPU-runtime compatibility and byte oracle check.

Builds a host shared library in --build. No writes to PSC-ONE, no FPGA/RTL runs.
"""
import argparse
import ctypes as c
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src/simulation'))
from int8_model import INT8Predictor, interpreter_api


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--os', type=Path, required=True, help='PSC-ONE/software/os directory, read only')
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--states', type=Path, required=True)
    p.add_argument('--expect-unsupported', action='store_true',
                   help='Assert PSC_TFLITE_ERR_UNSUPPORTED for the native LeakyReLU model')
    p.add_argument('--rollouts', action='store_true',
                   help='Also run the unchanged PSC CPU runtime in the same MuJoCo evaluation suite')
    a = p.parse_args()
    a.build.mkdir(parents=True, exist_ok=True)
    src = a.os / 'src/api/tflite'
    vendor = a.os / 'third_party/tflite'
    sources = [src / name for name in ('tflite_inspect.cc', 'tflite_runtime.cc',
                                     'tflite_quant.cc', 'tflite_synap.cc')]
    sources.append(a.os / 'tests/tflite/synap_mock.cc')
    library = a.build / 'psc_runtime.so'
    command = ['g++', '-shared', '-fPIC', '-std=c++17', '-O2', '-DTFLITE_SINGLE_ROUNDING=0',
               '-include', str(src / 'tflite_api.h')]
    for path in (src, vendor, vendor / 'flatbuffers/include', vendor / 'gemmlowp'):
        command.append('-I' + str(path))
    subprocess.run(command + [str(s) for s in sources] + ['-o', str(library)], check=True)
    lib = c.CDLL(str(library.resolve()))
    lib.psc_tflite_prepare.argtypes = [c.c_void_p, c.c_size_t]
    lib.psc_tflite_get_input.argtypes = [c.POINTER(c.c_size_t)]
    lib.psc_tflite_get_input.restype = c.POINTER(c.c_int8)
    lib.psc_tflite_get_output.argtypes = [c.POINTER(c.c_size_t)]
    lib.psc_tflite_get_output.restype = c.POINTER(c.c_int8)
    lib.psc_tflite_arena_used.restype = c.c_size_t
    model = a.model.read_bytes()
    storage = c.create_string_buffer(len(model) + 16)
    aligned = (c.addressof(storage) + 15) & ~15
    c.memmove(aligned, model, len(model))
    rc = lib.psc_tflite_prepare(aligned, len(model))
    result = dict(model_sha256=hashlib.sha256(model).hexdigest(), prepare_status=rc,
                  source_sha256={str(s.relative_to(a.os)): hashlib.sha256(s.read_bytes()).hexdigest()
                                 for s in sources}, scope='host CPU runtime only; no target hardware test')
    if rc == 0:
        predictor = INT8Predictor(a.model)
        Interpreter, resolver = interpreter_api()
        reference = Interpreter(model_content=model, num_threads=1,
                                experimental_op_resolver_type=resolver.BUILTIN_REF)
        reference.allocate_tensors()
        with np.load(a.states) as archive:
            x = archive['inputs']
        x = x[np.linspace(0, len(x) - 1, min(4096, len(x)), dtype=int)]
        codes = [predictor.encode(s) for s in x]
        # Include quantization-space boundaries and independent random vectors.
        codes += [np.full((1, 4), v, np.int8) for v in (-128, 0, 127)]
        codes += list(np.random.default_rng(91).integers(-128, 128, (1024, 1, 4), dtype=np.int8))
        length = c.c_size_t()
        inp = lib.psc_tflite_get_input(c.byref(length))
        assert length.value == 4
        mismatches, max_lsb, reference_mismatches = 0, 0, 0
        examples = []
        for code in codes:
            c.memmove(inp, code.ctypes.data, 4)
            assert lib.psc_tflite_invoke() == 0
            out = lib.psc_tflite_get_output(c.byref(length))
            assert length.value == 1
            predictor.interpreter.set_tensor(predictor.input['index'], code)
            predictor.interpreter.invoke()
            expected = int(predictor.interpreter.get_tensor(predictor.output['index'])[0, 0])
            reference.set_tensor(reference.get_input_details()[0]['index'], code)
            reference.invoke()
            reference_output = int(reference.get_tensor(reference.get_output_details()[0]['index'])[0, 0])
            reference_mismatches += out[0] != reference_output
            diff = abs(out[0] - expected)
            mismatches += diff != 0
            max_lsb = max(max_lsb, diff)
            if diff and len(examples) < 10:
                examples.append(dict(input=code.tolist(), psc=out[0], litert_optimized=expected,
                                     litert_reference=reference_output))
        result.update(samples=len(codes), mismatches=mismatches, maximum_error_lsb=max_lsb,
                      reference_mismatches=reference_mismatches, mismatch_examples=examples,
                      arena_bytes=lib.psc_tflite_arena_used())
        if a.rollouts:
            import mujoco
            from control_experiments import SCENE
            from int8_experiments import rollout, scenarios
            class PSCPredictor:
                def predict(self, state):
                    code = predictor.encode(state)
                    c.memmove(inp, code.ctypes.data, 4)
                    assert lib.psc_tflite_invoke() == 0
                    out = lib.psc_tflite_get_output(c.byref(length))
                    return ((out[0] - predictor.output_zero) * predictor.output_qscale
                            * predictor.output_scale + predictor.output_mean)
            physics = mujoco.MjModel.from_xml_path(str(SCENE))
            rows = []
            for scenario, torque in scenarios(20261009, 20, 6):
                row, _ = rollout(physics, scenario, torque, PSCPredictor())
                row['controller'] = 'PSC CPU host INT8'
                rows.append(row)
                print(f'PSC host {scenario.name}: fall={row["fell"]}, '
                      f'time={row["balancing_time_s"]:.3f}s', flush=True)
            result['rollouts'] = rows
    (a.build / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if a.expect_unsupported:
        assert rc == -202, f'Expected unsupported profile, got {rc}'
    else:
        assert rc == 0, f'PSC-ONE prepare failed: {rc}'
        assert result['mismatches'] == 0, 'PSC-ONE and LiteRT output bytes differ'


if __name__ == '__main__':
    main()
