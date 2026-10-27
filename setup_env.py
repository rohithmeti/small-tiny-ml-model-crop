#!/usr/bin/env python3
"""Build the Python environment for this project (CPU only). Uses only the Python standard library.

    python setup_env.py              # first-time setup: creates the folder .venv and installs everything into it
    python setup_env.py --recreate   # delete .venv and build it again

You normally do not run this by hand: START_HERE_windows.bat / start_linux_mac.sh do it when .venv is missing.
Needs internet the first time (about 350 MB of downloads). Works on Windows and Linux with Python 3.10 - 3.13.
The .venv folder belongs to the machine that built it: never copy it to another computer, just run this again there.
"""
import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
TORCH_VERSION = "2.14.1"
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
SUPPORTED = ((3, 10), (3, 13))
PIP_FLAGS = ["--disable-pip-version-check", "--progress-bar", "off", "--retries", "10", "--timeout", "60"]
IS_WIN = platform.system() == "Windows"
IS_MAC = platform.system() == "Darwin"


def venv_python():
    return VENV / ("Scripts/python.exe" if IS_WIN else "bin/python")


def child_env():
    """Keep the new environment isolated from PYTHONPATH (e.g. ROS) and from per-user packages."""
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONUTF8="1")
    env.pop("PYTHONPATH", None)
    return env


def run(cmd):
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, env=child_env())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recreate", action="store_true", help="delete .venv and rebuild it")
    a = ap.parse_args()

    v = sys.version_info[:2]
    print(f"Python {platform.python_version()} on {platform.platform()}")
    if not (SUPPORTED[0] <= v <= SUPPORTED[1]):
        sys.exit(f"\nThis Python ({v[0]}.{v[1]}) is not supported. Install Python 3.10, 3.11, 3.12 or 3.13 "
                 "(https://www.python.org/downloads/) and run this again with that version.")
    if platform.machine().lower() in ("arm64", "aarch64"):
        print("! ARM machine: most wheels exist, but this combination is untested.")

    if a.recreate and VENV.exists():
        shutil.rmtree(VENV)
    if not venv_python().exists():
        print(f"\n[1/4] creating virtual environment {VENV}")
        try:
            run([sys.executable, "-m", "venv", VENV])
        except subprocess.CalledProcessError:
            shutil.rmtree(VENV, ignore_errors=True)
            sys.exit("\nCould not create a virtual environment.\n"
                     "  Ubuntu/Debian: sudo apt install python3-venv\n"
                     "  Windows: re-run the Python installer, choose Modify, and tick 'pip' (venv is included).")
    py = venv_python()

    print("\n[2/4] upgrading pip")
    run([py, "-m", "pip", "install", "-U", "pip", *PIP_FLAGS])
    print("\n[3/4] installing the libraries from requirements.txt")
    run([py, "-m", "pip", "install", "-r", ROOT / "requirements.txt", *PIP_FLAGS])
    print(f"\n[4/4] installing PyTorch {TORCH_VERSION} (CPU build, about 200 MB)")
    cmd = [py, "-m", "pip", "install", f"torch=={TORCH_VERSION}", *PIP_FLAGS]
    if not IS_MAC:  # on macOS the normal PyPI wheel is already CPU/Metal
        cmd += ["--index-url", TORCH_CPU_INDEX]
    run(cmd)

    print("\nchecking the installation ...")
    rc = subprocess.run([str(py), str(ROOT / "src/check_env.py")], env=child_env()).returncode
    if rc:
        sys.exit(f"\nThe environment check FAILED (exit {rc}). Send the messages above to whoever maintains this project.")
    print("\nSetup finished. Next:  run.py demo   (or double-click START_HERE_windows.bat / run ./start_linux_mac.sh)")


if __name__ == "__main__":
    main()
