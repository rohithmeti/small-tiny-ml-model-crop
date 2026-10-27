"""End-to-end self test. Prints PASS/FAIL per check. Needs a trained model (run `train` first).

    python run.py test
"""
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from common import DATA_FILE, FEATURES, ROOT, add_sensor_noise, find_model_dir, get_device, load_data, split_data
from live_predict import parse_line

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  - {detail}" if detail else ""), flush=True)


def py(script, *args, timeout=120):
    env = dict(os.environ)
    return subprocess.run([sys.executable, str(ROOT / script), *args], capture_output=True, text=True, env=env, timeout=timeout)


# 1. everything runs on the CPU
import torch
check("PyTorch is the CPU build (no GPU needed)", get_device() == "cpu" and not torch.cuda.is_available(), f"torch {torch.__version__}")

# 2. dataset is the exact file we expect
want = json.loads((ROOT / "datasets/manifest.json").read_text())["datasets"][0]["sha256"]
check("primary dataset checksum", hashlib.sha256(DATA_FILE.read_bytes()).hexdigest() == want)

# 3. sensor-noise model behaves
X, y, classes = load_data()
noisy = add_sensor_noise(np.tile(X[:200], (50, 1)), "good", np.random.default_rng(0))
t_std = float((noisy[:, 3] - np.tile(X[:200, 3], 50)).std())
check("sensor noise: temperature error ~0.5 C (DHT22)", 0.4 < t_std < 0.6, f"measured {t_std:.2f}")
check("sensor noise keeps values physical", noisy[:, 4].max() <= 100 and noisy[:, 5].min() >= 0 and noisy[:, [0, 1, 2, 6]].min() >= 0)

# 4. saved model works and is accurate
mdir = find_model_dir()
meta = json.loads((mdir / "crop_model.json").read_text())
import onnxruntime as ort
sess = ort.InferenceSession(str(mdir / "crop_model.onnx"), providers=["CPUExecutionProvider"])
_, Xte, _, yte = split_data(X, y)
acc = float((sess.run(None, {meta["onnx_input_name"]: Xte})[0] == yte).mean())
check("saved ONNX model accuracy on the 440 test rows >= 97 %", acc >= 0.97, f"{acc:.4f} ({meta['model']}, from {mdir.name}/)")

# 5. reading validation
lo, hi = np.array(meta["input_min"]), np.array(meta["input_max"])
rain = float(np.median(X[:, 6]))
good = "N=90,P=42,K=43,T=21.5,H=80.1,PH=6.50,R=202.9"
check("accepts a normal reading", parse_line(good, lo, hi, rain)[0] is not None)
check("accepts a reading without rainfall (uses default)", parse_line(good.rsplit(",", 1)[0], lo, hi, rain)[2] is not None)
bad = {"garbage": "###garbage###", "NaN": "N=nan,P=1,K=1,T=20,H=80,PH=6,R=100", "missing field": "N=90,P=42,K=43,T=21.5,PH=6.5",
       "spike": "N=9999,P=42,K=43,T=21.5,H=80,PH=6.5,R=200", "pH 17": "N=90,P=42,K=43,T=21.5,H=80,PH=17,R=200",
       "humidity 115 %": "N=90,P=42,K=43,T=21.5,H=115,PH=6.5,R=200", "negative K": "N=90,P=42,K=-5,T=21.5,H=80,PH=6.5,R=200"}
for k, line in bad.items():
    check(f"rejects bad reading: {k}", parse_line(line, lo, hi, rain)[1] is not None)

# 6. whole pipeline over the network (what the ESP32 will do): sim -> TCP -> live
s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
live = subprocess.Popen([sys.executable, str(ROOT / "src/live_predict.py"), "--source", f"listen://127.0.0.1:{port}", "--count", "12"],
                        stdout=subprocess.PIPE, text=True, env=dict(os.environ))
time.sleep(1.5)
sim = py("sim/sensor_sim.py", "--out", f"tcp://127.0.0.1:{port}", "--count", "12", "--rate", "50", "--preset", "good")
out = live.communicate(timeout=60)[0]
check("sim -> TCP -> live: 12 readings arrive and are answered", sim.returncode == 0 and "summary: 12 lines, 12 used" in out, out.strip().splitlines()[-1] if out.strip() else "no output")

# 7. same over a (virtual) serial port, if socat is installed
if shutil.which("socat"):
    a, b = "/tmp/annu_test_tx", "/tmp/annu_test_rx"
    sc = subprocess.Popen(["socat", f"pty,raw,echo=0,link={a}", f"pty,raw,echo=0,link={b}"], stderr=subprocess.DEVNULL)
    time.sleep(1)
    live = subprocess.Popen([sys.executable, str(ROOT / "src/live_predict.py"), "--source", f"serial:{b}", "--count", "8"],
                            stdout=subprocess.PIPE, text=True, env=dict(os.environ))
    time.sleep(1.5)
    sim = py("sim/sensor_sim.py", "--out", f"serial:{a}", "--count", "8", "--rate", "50")
    try:
        out = live.communicate(timeout=30)[0]
    except subprocess.TimeoutExpired:
        live.kill(); out = ""
    sc.terminate()
    check("sim -> virtual serial port -> live", "summary: 8 lines, 8 used" in out, out.strip().splitlines()[-1] if out.strip() else "no output")
else:
    print("SKIP  serial test (install socat to enable: sudo apt install socat)")

# 8. accuracy in the live demo (good sensors, no glitches), noise-aware model should stay above 90 %
r = py("src/live_predict.py", "--source", "demo", "--preset", "good", "--glitch", "0", "--count", "110")
m = re.search(r"(\d+)/(\d+) correct", r.stdout)
check("live demo with 'good' sensors: >= 90 % correct", bool(m) and int(m.group(1)) / int(m.group(2)) >= 0.90, m.group(0) if m else r.stdout[-200:])

# 9. the dashboard serves the page and live data
import urllib.request
s = socket.socket(); s.bind(("127.0.0.1", 0)); dport = s.getsockname()[1]; s.close()
dash = subprocess.Popen([sys.executable, str(ROOT / "src/dashboard.py"), "--no-browser", "--port", str(dport), "--rate", "8"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=dict(os.environ))
state, page, more = None, "", {}
try:
    for _ in range(40):
        time.sleep(0.5)
        try:
            state = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{dport}/api/state", timeout=3).read())
            if state["used"] >= 4:
                break
        except Exception:  # noqa: BLE001 - server still starting
            pass
    page = urllib.request.urlopen(f"http://127.0.0.1:{dport}/", timeout=3).read().decode()
    get = lambda path: json.loads(urllib.request.urlopen(f"http://127.0.0.1:{dport}{path}", timeout=5).read())  # noqa: E731
    more["rows"] = get("/api/dataset?crop=rice&size=5")
    more["a"] = get("/api/predict?N=90&P=42&K=43&T=20.9&H=82&PH=6.5&R=202.9")
    more["b"] = get("/api/predict?N=140&P=42&K=140&T=20.9&H=82&PH=6.5&R=202.9")
    more["bad"] = get("/api/predict?N=90&P=42&K=43&T=20.9&H=150&PH=6.5&R=202.9")
finally:
    dash.terminate()
check("dashboard serves the page and live readings", bool(state) and state["used"] >= 4 and "Crop advisor" in page and "const INITIAL = {" in page,
      f"{state['used']} readings, latest crop {state['latest']['crop']}" if state and state.get("latest") else "no data")
check("dashboard data tab + playground: dataset rows, what-if answers, bad value refused",
      bool(more) and more["rows"]["total"] == 100 and len(more["rows"]["rows"]) == 5 and more["rows"]["rows"][0]["line"].startswith("N=")
      and more["a"]["ok"] and more["b"]["ok"] and more["a"]["top"] != more["b"]["top"] and more["bad"]["ok"] is False,
      f"changing N,K moved the answer {more['a']['top'][0]['crop']} -> {more['b']['top'][0]['crop']}" if more else "no data")

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
