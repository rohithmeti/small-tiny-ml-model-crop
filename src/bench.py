"""Speed test of the saved model. Answers: 'is it fast enough on a CPU-only laptop?'

    python run.py bench

INPUT : models/crop_model.onnx  (+ crop_mlp.pth)   OUTPUT: printed table + results/bench.json
"""
import json
import time

import numpy as np

from common import FEATURES, TorchMLP, find_model_dir, get_device, load_data, out_dir, split_data

import onnxruntime as ort

mdir = find_model_dir()
meta = json.loads((mdir / "crop_model.json").read_text())
X, y, classes = load_data()
_, Xte, _, yte = split_data(X, y)
print(f"model: {meta['model']} (noise-aware={meta['noise_aware']}) from {mdir.name}/   compute mode: {get_device()}")

sess = ort.InferenceSession(str(mdir / "crop_model.onnx"), providers=["CPUExecutionProvider"])
name = sess.get_inputs()[0].name
pred = sess.run(None, {name: Xte})[0]
res = {"model": meta["model"], "onnx_test_accuracy": float((pred == yte).mean()), "onnx_kb": round((mdir / "crop_model.onnx").stat().st_size / 1024, 1)}
print(f"ONNX on the 440 test rows: accuracy {res['onnx_test_accuracy']:.4f}, file {res['onnx_kb']} KB")


def timeit(fn, n):
    fn()
    t = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(t)), float(np.percentile(t, 95))


one = Xte[:1]
med, p95 = timeit(lambda: sess.run(None, {name: one}), 2000)
res["onnx_cpu_1row_ms"] = {"median": med, "p95": p95}
big = np.tile(Xte, (23, 1))[:10000]
med_b, _ = timeit(lambda: sess.run(None, {name: big}), 20)
res["onnx_cpu_10000rows_ms"] = med_b
print(f"ONNX Runtime on CPU : 1 reading {med:.3f} ms (p95 {p95:.3f}) | 10,000 readings {med_b:.1f} ms = {10000 / med_b * 1e3:,.0f} readings/s")

# the PyTorch network (.pth), also on the CPU
pth = mdir / "crop_mlp.pth"
if pth.exists():
    m = TorchMLP.load_pth(pth)
    m1, _ = timeit(lambda: m.predict_proba(one), 500)
    mb, _ = timeit(lambda: m.predict_proba(big), 20)
    acc = float((m.predict(Xte) == yte).mean())
    res["torch_cpu"] = {"1row_ms": m1, "10000rows_ms": mb, "test_acc": acc}
    print(f"PyTorch .pth on CPU : 1 reading {m1:.3f} ms | 10,000 readings {mb:.1f} ms | test accuracy {acc:.4f}")

(out_dir("results") / "bench.json").write_text(json.dumps(res, indent=2))
print("Take-away: the model has only 7 inputs, so a CPU answers in well under a millisecond - no GPU needed.")
