# How to run the crop model (simple steps)

Open a terminal **inside this folder** first:

```
cd ~/Downloads/annu_ml_cpu
```

The first run builds the Python environment by itself (needs internet, a few minutes). After that it starts at once.
Windows: use `START_HERE_windows.bat` instead of `./start_linux_mac.sh` in every command below.

---

## Way 1: with the dashboard (easiest)

```
./start_linux_mac.sh dashboard
```

The browser opens at **http://127.0.0.1:8050** (if not, open that address yourself).

| Tab | What you see |
|---|---|
| **Live** | recommended crop, the 7 sensor values, and the live confusion matrix filling up |
| **Models** | all 12 models: results table + confusion matrices |
| **Data** | the 2200 training rows |
| **Playground** | move sliders and watch the answer change |

Stop it: press **Ctrl + C** in the terminal.

---

## Way 2: by hand, in the terminal

Run these one by one:

| Step | Command | What it does |
|---|---|---|
| 1 | `./start_linux_mac.sh check` | tests that everything is installed |
| 2 | `./start_linux_mac.sh train` | trains the models (about 1 minute), saves the best one in `models/` |
| 3 | `./start_linux_mac.sh demo` | gives the model 30 fake sensor readings and prints the crop for each |
| 4 | `./start_linux_mac.sh test` | checks the whole project (should say 19/19 passed) |

**With a sensor sending data (2 terminals):**

Terminal 1 (waits for data):
```
./start_linux_mac.sh live --source listen://0.0.0.0:5555
```

Terminal 2 (fake ESP32 sends readings):
```
./start_linux_mac.sh sim --out tcp://127.0.0.1:5555
```

To see this in the dashboard instead, put `dashboard` in place of `live` in terminal 1.

Stop anything: **Ctrl + C**.
