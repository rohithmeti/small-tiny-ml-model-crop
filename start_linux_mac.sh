#!/usr/bin/env sh
# Crop recommendation (CPU only).  First run builds the Python environment (.venv, needs internet);
# next runs start immediately.   ./start_linux_mac.sh            -> demo
#                                ./start_linux_mac.sh train      -> any step of run.py (see README.md)
cd "$(dirname "$0")" || exit 1

PY=""
for c in python3.12 python3.11 python3.13 python3.10 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if (3,10)<=sys.version_info[:2]<=(3,13) else 1)' 2>/dev/null; then
    PY="$c"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.10 - 3.13 not found. Install it (Ubuntu: sudo apt install python3 python3-venv) and run this again."
  exit 1
fi
echo "Using Python: $PY ($($PY --version 2>&1))"

if [ ! -x ".venv/bin/python" ]; then
  echo; echo "First run: building the environment (downloads about 350 MB, takes a few minutes)."
  "$PY" setup_env.py || { echo "Setup failed - see the messages above."; exit 1; }
fi

if [ $# -eq 0 ]; then exec .venv/bin/python run.py demo; else exec .venv/bin/python run.py "$@"; fi
