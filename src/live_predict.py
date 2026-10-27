"""Laptop side of the IoT pipeline: receive sensor lines, check them, print crop recommendations.

    python run.py live --source demo                     # everything in one process (built-in fake sensor)
    python run.py live --source listen://0.0.0.0:5555    # wait for the ESP32 / sim to connect over Wi-Fi (TCP)
    python run.py live --source serial:/dev/ttyUSB0      # read a serial port
    python run.py live --source stdin                    # e.g.  python run.py sim --out stdout | python run.py live --source stdin

INPUT : lines like  N=90,P=42,K=43,T=21.5,H=80.1,PH=6.50,R=202.9      (R = rainfall in mm; if missing, a typical value is used)
OUTPUT: one line per reading:  #3 rice 94% | top3: rice 94, jute 3, maize 2 | smoothed: rice 91% | 0.04 ms
        bad readings are printed as REJECTED with the reason and never reach the model.
        The model is models/crop_model.onnx, always run on the CPU with ONNX Runtime.
"""
import argparse
import collections
import json
import math
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim"))
from common import FEATURES, find_model_dir, load_data  # noqa: E402

KEYS = {"N": 0, "P": 1, "K": 2, "T": 3, "H": 4, "PH": 5, "R": 6}  # line key -> position in FEATURES
# hard physical limits (min, max): nothing outside these can be a real measurement
PHYSICAL = [(0, np.inf)] * 3 + [(-40, 60), (0, 100), (0, 14), (0, np.inf)]


def parse_line(line, lo, hi, rain_default):
    """-> (feature_vector or None, problem_text or None, note_text or None)."""
    vals, margin = {}, 0.25 * (hi - lo)
    try:
        for part in line.strip().split(","):
            k, v = part.split("=")
            vals[KEYS[k.strip().upper()]] = float(v)
    except (ValueError, KeyError):
        return None, f"cannot parse {line.strip()[:50]!r}", None
    note = None
    if 6 not in vals:
        vals[6], note = rain_default, f"no rainfall given, assuming {rain_default:.0f} mm"
    missing = [FEATURES[i] for i in range(7) if i not in vals]
    if missing:
        return None, f"missing {','.join(missing)}", None
    x = np.array([vals[i] for i in range(7)], dtype=np.float32)
    for i, v in enumerate(x):
        if not math.isfinite(v):
            return None, f"{FEATURES[i]} is not a number", None
        low, high = max(lo[i] - margin[i], PHYSICAL[i][0]), min(hi[i] + margin[i], PHYSICAL[i][1])
        if not (low <= v <= high):
            return None, f"{FEATURES[i]}={v:g} outside plausible range {low:.1f}..{high:.1f}", None
    return x, None, note


def line_source(spec, preset, glitch, delay=0.2, hold=5):
    """Yield (line, true_crop_or_None) from the chosen source. delay/hold only matter for the demo source."""
    if spec == "demo":
        from sensor_sim import readings
        for truth, line in readings(preset, None, glitch, hold=hold):
            yield line, truth
            time.sleep(delay)
    elif spec == "stdin":
        for line in sys.stdin:
            yield line, None
    elif spec.startswith("serial:"):
        import serial
        ser = serial.Serial(spec[7:], 115200, timeout=1)
        while True:
            line = ser.readline().decode(errors="replace")
            if line.strip():
                yield line, None
    elif spec.startswith(("listen://", "tcp://")):
        host, port = spec.split("://")[1].rsplit(":", 1)
        if spec.startswith("listen://"):
            srv = socket.socket()
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((host, int(port)))
            srv.listen(1)
            print(f"listening on {host}:{port} ... (start the sensor / `python run.py sim --out tcp://127.0.0.1:{port}`)", flush=True)
            while True:
                conn, addr = srv.accept()
                print(f"sensor connected from {addr[0]}", flush=True)
                with conn, conn.makefile("r", errors="replace") as f:
                    for line in f:
                        yield line, None
                print("sensor disconnected; waiting for it to reconnect", flush=True)
        else:
            with socket.create_connection((host, int(port))) as conn, conn.makefile("r", errors="replace") as f:
                for line in f:
                    yield line, None
    else:
        sys.exit("--source must be demo | stdin | serial:/dev/xxx | listen://host:port | tcp://host:port")


class LivePredictor:
    """Loads the saved model and turns one text line into a recommendation (used by this script and by dashboard.py)."""

    def __init__(self, smooth=5):
        import onnxruntime as ort
        self.model_dir = find_model_dir()
        self.meta = json.loads((self.model_dir / "crop_model.json").read_text())
        self.sess = ort.InferenceSession(str(self.model_dir / "crop_model.onnx"), providers=["CPUExecutionProvider"])
        self.in_name, self.classes = self.meta["onnx_input_name"], self.meta["classes"]
        self.lo, self.hi = np.array(self.meta["input_min"]), np.array(self.meta["input_max"])
        X, _, _ = load_data()
        self.rain_default = float(np.median(X[:, 6]))
        self.recent = collections.deque(maxlen=max(smooth, 1))  # last few probability vectors, for smoothing

    def process(self, line, remember=True):
        """remember=False answers without touching the smoothing buffer (used for "what if" questions).
        Rejected line -> {"ok": False, "problem": text}.
        Good line -> {"ok": True, "x": the 7 values, "p": probabilities, "smoothed": averaged probabilities, "ms": model time, "note": text or None}."""
        x, problem, note = parse_line(line, self.lo, self.hi, self.rain_default)
        if problem:
            return {"ok": False, "problem": problem}
        t0 = time.perf_counter()
        p = self.sess.run(None, {self.in_name: x[None, :]})[1][0]
        ms = (time.perf_counter() - t0) * 1e3
        if not remember:
            return {"ok": True, "x": x, "p": p, "smoothed": p, "ms": ms, "note": note}
        self.recent.append(p)
        return {"ok": True, "x": x, "p": p, "smoothed": np.mean(self.recent, axis=0), "ms": ms, "note": note}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="demo")
    ap.add_argument("--preset", default="good", choices=["clean", "good", "cheap"], help="demo source only: sensor quality")
    ap.add_argument("--glitch", type=float, default=0.1, help="demo source only: fraction of broken readings")
    ap.add_argument("--smooth", type=int, default=5, help="average the probabilities of the last N good readings")
    ap.add_argument("--count", type=int, default=0, help="stop after N lines (0 = until Ctrl-C)")
    a = ap.parse_args()

    pred = LivePredictor(a.smooth)
    classes = pred.classes
    print(f"model: {pred.meta['model']} (noise-aware={pred.meta['noise_aware']}) from {pred.model_dir.name}/, running on CPU via ONNX Runtime")

    seen = ok = bad = right = known = 0
    ms = []
    try:
        for line, truth in line_source(a.source, a.preset, a.glitch):
            seen += 1
            r = pred.process(line)
            if not r["ok"]:
                bad += 1
                print(f"#{seen} REJECTED: {r['problem']}")
            else:
                p, sm = r["p"], r["smoothed"]
                ms.append(r["ms"])
                ok += 1
                top = np.argsort(-p)[:3]
                s = int(sm.argmax())
                msg = (f"#{seen} {classes[top[0]]} {p[top[0]]:.0%} | top3: " + ", ".join(f"{classes[i]} {p[i]:.0%}" for i in top)
                       + f" | smoothed: {classes[s]} {sm[s]:.0%} | {r['ms']:.2f} ms")
                if truth:
                    known += 1
                    right += classes[top[0]] == truth
                    msg += f" | truth: {truth} {'OK' if classes[top[0]] == truth else 'WRONG'}"
                print(msg + (f"  [{r['note']}]" if r["note"] else ""))
            if a.count and seen >= a.count:
                break
    except KeyboardInterrupt:
        pass
    print(f"\nsummary: {seen} lines, {ok} used, {bad} rejected" +
          (f", {right}/{known} correct ({right / known:.0%}) against the simulator's truth" if known else "") +
          (f", mean model time {np.mean(ms):.3f} ms" if ms else ""))


if __name__ == "__main__":
    main()
