# Crop recommendation - CPU version

> **New here?** This page = how to install and start. **`HOW_IT_WORKS.md` = what goes in and out, how the data is read, how it is trained and run.**

Give it 7 soil/weather numbers, it tells you which of 22 crops fits best. Runs on any laptop **without a GPU**
(Windows, Linux). Today the sensor data is simulated on the laptop; later an ESP32 will send the same numbers over Wi-Fi.

```
 7 numbers  ->  [ validate ]  ->  [ model: ONNX, CPU ]  ->  "rice 94%  | top3: rice, jute, maize"
 N P K temperature humidity pH rainfall
```

## 1. Start (first time needs internet, ~350 MB download, a few minutes)

| Computer | What to do |
|---|---|
| **Windows** | Install **Python 3.10 - 3.13** from python.org (tick *"Add python.exe to PATH"*). Unzip this folder somewhere short like `C:\annu_ml_cpu`. **Double-click `START_HERE_windows.bat`.** |
| **Linux / macOS** | Have Python 3.10 - 3.13 (`sudo apt install python3 python3-venv` on Ubuntu). In this folder run `./start_linux_mac.sh` |

The first run builds a private Python environment in the folder `.venv` (all libraries are installed *into it*, nothing
touches the rest of your system), then runs a 30-reading demo. Next runs skip the build and start at once.
**Never copy `.venv` to another computer** - it only works where it was built. Copy the folder without it, the launcher rebuilds it.

Other things to run (Windows: `START_HERE_windows.bat <step>`, Linux/macOS: `./start_linux_mac.sh <step>`):

| Step | What it does | Time |
|---|---|---|
| `demo` (default) | fake sensor -> model -> recommendations on screen | 10 s |
| `dashboard` | **dashboard in your web browser** with 3 tabs: *Live*, *Data* (browse the 2,200 training rows) and *Playground* (change N, K ... with sliders). Fake sensor by default; opens `http://127.0.0.1:8050` | until you press Ctrl+C |
| `train` | trains 6 model types x 2 ways, compares them, saves the best | ~1 min |
| `bench` | speed test of the saved model | 5 s |
| `test` | 19 automatic checks of the whole pipeline | < 1 min |
| `all` | check + train + bench + test | ~2 min |
| `data` | download the optional extra datasets | seconds |

## 2. What every file does

```
annu_ml_cpu/
├─ HOW_IT_WORKS.md          inputs, outputs, data, training and running explained step by step
├─ START_HERE_windows.bat   double-click launcher for Windows (finds Python, builds .venv once, runs a step)
├─ start_linux_mac.sh       same for Linux / macOS
├─ setup_env.py             builds .venv and installs requirements.txt + CPU PyTorch (called by the launchers)
├─ run.py                   runs one step inside .venv:  python run.py <step>
├─ requirements.txt         the exact library versions (pinned, so every laptop gets the same thing)
├─ src/
│  ├─ common.py             shared helpers: load data, sensor-noise model, small neural network
│  ├─ train.py              trains + compares models, picks one, saves it
│  ├─ bench.py              measures speed and accuracy of the saved model
│  ├─ live_predict.py       laptop side of the IoT link: receives sensor lines, checks them, predicts
│  ├─ dashboard.py          web server for the dashboard (reads the same sensor lines, keeps the latest state)
│  ├─ dashboard.html        the dashboard page itself (look and layout; edit it and refresh the browser)
│  ├─ check_env.py          "is the installation healthy?"
│  └─ selftest.py           19 end-to-end checks
├─ sim/sensor_sim.py        fake ESP32: produces realistic sensor lines and sends them
├─ datasets/                data (+ fetch_datasets.py, manifest.json = sources, licenses, checksums; DATASETS.md = index)
├─ models/                  the trained model (made by `train`, already included)
└─ results/                 report.md, confusion_matrix.png, metrics.json, bench.json (made by `train` / `bench`)
```

| File | Input | Output |
|---|---|---|
| `train.py` | `datasets/.../Crop_recommendation.csv` (2200 rows: 7 numbers + crop name) | `models/crop_model.onnx` (3 KB), `crop_model.json`, `crop_mlp.pth`; `results/report.md`, `confusion_matrix.png` |
| `bench.py` | `models/crop_model.onnx` | speed/accuracy printed + `results/bench.json` |
| `sensor_sim.py` | the dataset (real rows + simulated sensor error) | text lines: `N=90,P=42,K=43,T=21.5,H=80.1,PH=6.50,R=202.9` over TCP, serial port, or screen |
| `live_predict.py` | those text lines (from `demo`, TCP, serial port or stdin) | one answer per line, or `REJECTED: reason` |
| `dashboard.py` + `dashboard.html` | the same text lines (Live tab), the dataset file (Data tab), values you type or slide (Playground tab) | a web page: recommended crop, the raw text lines, 7 sensor tiles, the dataset table, a "what if" answer |

## 3. How it works

1. **Data**: 2200 field records, 22 crops x 100 each (Kaggle *Crop Recommendation Dataset*, Atharva Ingle; see `datasets/DATASETS.md`).
2. **Train** (`train.py`): 1760 rows to learn/tune, 440 rows locked away as the final exam. Six model types are tried
   (Naive Bayes, SVM, Random Forest, LightGBM, XGBoost, small neural net). Each is trained **twice**: normally, and
   *noise-aware* - with extra copies of the training rows that have realistic sensor error added. The model is chosen by
   5-fold cross-validation on the training rows only; the 440 exam rows are only used to report the score.
3. **Save**: the chosen model is written as an **ONNX** file - a portable format that any CPU runs through ONNX Runtime
   (also the format Edge Impulse accepts, for the ESP32 step later). The small PyTorch network is also saved as `.pth`.
4. **Live** (`live_predict.py`): each incoming line is parsed and range-checked (missing field, NaN, absurd value,
   pH 17 ... are rejected and never reach the model); good readings go to the model; the last 5 answers are averaged so
   the recommendation does not flicker.

## 4. How to handle it

* **Sensor side, today (simulation)** - in one window `python run.py live --source listen://127.0.0.1:5555`,
  in another `python run.py sim --out tcp://127.0.0.1:5555 --preset cheap --glitch 0.1`
  (Linux: `python3`; Windows: allow Python through the firewall if asked, *private networks* only).
  Serial port instead: `--source serial:COM3` (Windows) / `serial:/dev/ttyUSB0` (Linux), and `sim --out serial:...`.
* **Sensor side, later (ESP32)** - the ESP32 only has to connect to `laptop-ip:5555` and send the same one-line text
  every second. The laptop code does not change. Sensor quality presets: `clean`, `good` (DHT22 + decent NPK/pH), `cheap` (DHT11 + budget NPK/pH).
* **No rain sensor?** Leave out `R=...`; the laptop assumes a typical rainfall and the answer is less certain (see accuracy below).
* **Moving to another laptop**: copy the folder **without `.venv`**, run the launcher. Needs internet once.
  Keep the path short on Windows (long paths can break PyTorch's install).
* **Dashboard**: `python run.py dashboard` (Windows: `START_HERE_windows.bat dashboard`) opens a page in your browser. It takes the same sources as `live`, e.g.
  `python run.py dashboard --source listen://0.0.0.0:5555` for the ESP32; add `--host 0.0.0.0` to open it from a phone on the same Wi-Fi
  (`http://<laptop-ip>:8050`). Light or dark follows your computer; force one with `?theme=light` / `?theme=dark` on the address.
  **Data tab** = the training file as a table (filter by crop, page through it, each row shown as the text line a sensor would send).
  **Playground tab** = sliders and number boxes for all 7 inputs; raise N and K and the recommendation changes at once, with a list of what you changed.
* **Retrain**: `train` (any time; it overwrites `models/`).

| Problem | Fix |
|---|---|
| "Python 3.10 - 3.13 was not found" | install Python from python.org (3.14 is too new for the pinned libraries) |
| venv creation fails on Ubuntu | `sudo apt install python3-venv` |
| pip download errors / timeouts | check internet, run the launcher again (it resumes), or `python setup_env.py --recreate` |
| `live` waits forever | start `sim` / the ESP32 after `live` says *listening*; check the firewall and IP address |
| antivirus blocks `.bat` | run `python setup_env.py` then `.venv\Scripts\python run.py demo` in a terminal |

## 5. How well does it perform (measured, 440 unseen test rows)

| Situation | Accuracy |
|---|---|
| Perfect sensors (the dataset as is) | **99.1 %** (436 / 440; errors: 2 maize->cotton, 2 rice->jute) |
| Good sensors (DHT22 +-0.5 C, 3 % RH; NPK 5 %; pH 0.1; rain 10 %) | **98.0 %** |
| Cheap sensors (DHT11 +-2 C, 5 % RH; NPK 10 %; pH 0.3; rain 20 %) | **94.3 %** |
| Right crop within the top 3 suggestions | **100 %** (perfect / good sensors), **99.9 %** (cheap sensors) |
| No rainfall input at all | 94.3 % (perfect sensors), 89.5 % (cheap sensors) |

* The deployed model is **Gaussian Naive Bayes, noise-aware**. Random Forest and the neural network score within noise of it
  (see `results/report.md`) - with 440 test rows one wrong answer is 0.23 %, so don't read the ranking too finely.
* **Noise-aware training is what matters for real sensors**: without it, accuracy with cheap sensors falls to 86 % (Naive Bayes),
  78 % (LightGBM), 77 % (XGBoost); with it, all six model types reach 92.8 - 94.3 %.
* Speed (laptop CPU, ONNX Runtime): **0.014 ms per reading**, ~8.8 million readings/s in batches. Model file 3 KB. A GPU gives nothing at this size.
* Disk: `.venv` ~1.4 GB. Whole pipeline `all`: ~2 min.

**Be honest about the limits.** The dataset is very clean (features inside one crop are almost uncorrelated: it looks synthetic),
so ~99 % on it is expected and published papers report the same. The "good/cheap sensor" rows are *simulated* error
from datasheet figures (DHT22/DHT11 from Adafruit's guide, NPK "<=5 %" from the DFRobot SEN0605 page; the pH and rainfall
figures are assumptions), not real measurements - real soil NPK sensors are often worse. The model recommends the crop whose
typical conditions match; it knows nothing about price, season, or local soil rules.

## 6. Credits and tested status

* Data: Atharva Ingle, *Crop Recommendation Dataset* (Kaggle; a CC BY 4.0 copy exists on Figshare). Other datasets
  and their licenses: `datasets/DATASETS.md`.
* Tested on Ubuntu 22.04 / Python 3.10 from a fresh copy of this folder (launcher -> environment build -> train -> bench -> 19/19 self-tests).
  **Not run on Windows yet**: the launcher and code are Windows-aware, and every pinned library, plus PyTorch CPU, has a Windows wheel
  for Python 3.10 - 3.13 (checked on PyPI / the PyTorch index), but you are the first to run it there - if something fails, send the message printed in the window.
