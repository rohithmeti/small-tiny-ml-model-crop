"""Fake ESP32. Takes real rows from the dataset, adds realistic sensor error, sends one text line per reading.

    python run.py sim --out tcp://127.0.0.1:5555     # connect to the laptop's listener (the ESP32 will do exactly this over Wi-Fi)
    python run.py sim --out serial:/tmp/annu_tx      # write to a serial port (real USB serial, or a virtual one made with socat)
    python run.py sim --out stdout                   # just print

    --preset good|cheap|clean   how bad the sensors are (see src/common.py)     --rate 2     readings per second
    --crop rice                 stay on one crop (default: cycle through all 22, 5 readings each)
    --glitch 0.1                fraction of broken readings (missing field, NaN, spike, garbage) to test the validation
    --no-rain                   leave out the rainfall field (a real ESP32 has no rain sensor)
    --count 50                  stop after 50 readings

Line format (also what the ESP32 firmware should print):   N=90,P=42,K=43,T=21.5,H=80.1,PH=6.50,R=202.9
"""
import argparse
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from common import SEED, add_sensor_noise, load_data  # noqa: E402


def format_line(row, with_rain=True):
    n, p, k, t, h, ph, r = (float(v) for v in row)
    s = f"N={n:.0f},P={p:.0f},K={k:.0f},T={t:.1f},H={h:.1f},PH={ph:.2f}"
    return s + (f",R={r:.1f}" if with_rain else "")


def glitch(line, rng):
    kind = rng.integers(4)
    parts = line.split(",")
    if kind == 0 and len(parts) > 2:
        del parts[rng.integers(len(parts))]                       # a field is missing
    elif kind == 1:
        parts[rng.integers(len(parts))] = parts[0].split("=")[0] + "=nan"  # NaN
    elif kind == 2:
        parts[0] = "N=9999"                                       # absurd spike
    else:
        return "###garbage###"
    return ",".join(parts)


def readings(preset="good", crop=None, glitch_rate=0.0, with_rain=True, seed=SEED, hold=5):
    """Endless generator of (true_crop_name, text_line); stays on one crop for `hold` readings, then moves to the next."""
    X, y, classes = load_data()
    rng = np.random.default_rng(seed)
    order = [classes.index(crop)] if crop else list(rng.permutation(len(classes)))
    i = 0
    while True:
        c = order[(i // hold) % len(order)] if not crop else order[0]
        row = X[rng.choice(np.where(y == c)[0])][None, :]
        line = format_line(add_sensor_noise(row, preset, rng)[0], with_rain)
        if rng.random() < glitch_rate:
            line = glitch(line, rng)
        yield classes[c], line
        i += 1


def open_output(spec):
    """-> function(line) that sends one line."""
    if spec == "stdout":
        return lambda line: print(line, flush=True)
    if spec.startswith("tcp://"):
        host, port = spec[6:].rsplit(":", 1)
        for _ in range(60):  # wait up to 30 s for the laptop's listener
            try:
                s = socket.create_connection((host, int(port)), timeout=5)
                break
            except OSError:
                time.sleep(0.5)
        else:
            sys.exit(f"could not connect to {spec} (is `python run.py live --source listen://0.0.0.0:{port}` running?)")
        return lambda line: s.sendall((line + "\n").encode())
    if spec.startswith("serial:"):
        import serial
        ser = serial.Serial(spec[7:], 115200, timeout=1)
        return lambda line: ser.write((line + "\n").encode())
    sys.exit("--out must be tcp://host:port, serial:/dev/xxx or stdout")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="stdout")
    ap.add_argument("--preset", default="good", choices=["clean", "good", "cheap"])
    ap.add_argument("--crop")
    ap.add_argument("--rate", type=float, default=2.0)
    ap.add_argument("--count", type=int, default=0, help="0 = run until Ctrl-C")
    ap.add_argument("--glitch", type=float, default=0.0)
    ap.add_argument("--no-rain", action="store_true")
    a = ap.parse_args()
    send = open_output(a.out)
    try:
        for n, (truth, line) in enumerate(readings(a.preset, a.crop, a.glitch, not a.no_rain), 1):
            send(line)
            print(f"[sim] #{n} (true crop: {truth}) -> {line}", file=sys.stderr, flush=True) if a.out != "stdout" else None
            if a.count and n >= a.count:
                break
            time.sleep(1.0 / a.rate)
    except (BrokenPipeError, ConnectionResetError, KeyboardInterrupt):
        pass
