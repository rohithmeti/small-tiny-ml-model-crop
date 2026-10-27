#!/usr/bin/env python3
"""One launcher for everything (Windows, Linux, macOS).  Uses only the Python standard library.

    python run.py <step> [extra arguments for that step]

    steps:  data    download + verify the optional extra datasets (the main one is already included)
            check   test that the environment works
            train   train + compare models, save the best one as ONNX      (about 1 minute)
            bench   speed test of the saved model
            demo    laptop-side demo with a built-in fake sensor            (30 readings)
            dashboard  live dashboard in your web browser (built-in fake sensor, or --source listen://0.0.0.0:5555)
            live    receive real/simulated sensor lines and print recommendations  (see README)
            sim     fake ESP32: send simulated sensor lines to `live`
            test    end-to-end self test
            all     check + train + bench + test

Runs inside the .venv created by setup_env.py.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STEPS = {"check": "src/check_env.py", "train": "src/train.py", "bench": "src/bench.py",
         "sim": "sim/sensor_sim.py", "live": "src/live_predict.py", "dashboard": "src/dashboard.py", "test": "src/selftest.py"}


def venv_python():
    p = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not p.exists():
        sys.exit("The environment is not built yet. Run:  python setup_env.py   (or START_HERE_windows.bat / ./start_linux_mac.sh)")
    return p


def run_step(step, extra):
    env = dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg", PYTHONNOUSERSITE="1", PYTHONUTF8="1")
    env.pop("PYTHONPATH", None)  # keep the venv isolated from other Python setups (e.g. ROS)
    cmd = [str(venv_python()), str(ROOT / STEPS[step])] + extra
    print(f"\n>>> {step}: {Path(cmd[1]).name} {' '.join(extra)}", flush=True)
    return subprocess.run(cmd, env=env, cwd=ROOT).returncode


def main():
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    step, extra = args[0], args[1:]
    if step == "data":
        return subprocess.run([sys.executable, str(ROOT / "datasets/fetch_datasets.py")] + extra).returncode
    if step == "demo":
        step, extra = "live", ["--source", "demo", "--count", "30"] + extra
    if step == "all":
        for s in ("check", "train", "bench", "test"):
            rc = run_step(s, [])
            if rc:
                print(f"\nstep '{s}' failed (exit {rc}) - stopping.")
                return rc
        return 0
    if step not in STEPS:
        sys.exit(f"unknown step '{step}'. Steps: data, demo, all, {', '.join(STEPS)}")
    return run_step(step, extra)


if __name__ == "__main__":
    sys.exit(main())
