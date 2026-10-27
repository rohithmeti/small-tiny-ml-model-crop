# How it works

This file explains the project from the inside: what goes in, what comes out, how the data is read,
how the model is trained, and how it is run. (For *installing and starting* it, read `README.md`.)

---

## 1. The whole thing in one picture

```
 TRAINING - done once:  python run.py train           RUNNING - for every sensor reading
 ------------------------------------------           ----------------------------------------------
 datasets/.../Crop_recommendation.csv                 text line from the ESP32 (or the simulator):
   2200 rows = 7 numbers + the right crop               N=90,P=42,K=43,T=21.5,H=80.1,PH=6.5,R=202.9
            |                                                         |
  split: 1760 rows to learn, 440 rows kept            src/live_predict.py  -> read the line, check it
  hidden as an "exam"                                                 |
            |                                         models/crop_model.onnx  (run by ONNX Runtime, CPU)
  try 6 kinds of model, each 2 ways                                   |
            |                                         22 probabilities -> top 3 -> average of last 5
  pick the best one                                                   |
            |                                         #1 rice 64% | top3: rice 64%, jute 36%, coffee 0%
  save it:  models/crop_model.onnx
```

---

## 2. Input and output

**The model itself** (this is the heart of the project):

| | Content |
|---|---|
| **Input** | 7 numbers, always in this order: `N, P, K, temperature, humidity, pH, rainfall` |
| **Output** | 22 probabilities that add up to 1, one per crop (alphabetical: apple, banana, blackgram, chickpea, coconut, coffee, cotton, grapes, jute, kidneybeans, lentil, maize, mango, mothbeans, mungbean, muskmelon, orange, papaya, pigeonpeas, pomegranate, rice, watermelon). The recommendation is the biggest one; the top 3 are shown too. |

**What the 7 numbers mean and what values the training data contains:**

| Input | Meaning | Lowest | Average | Highest |
|---|---|---|---|---|
| N | nitrogen level in the soil | 0 | 50.6 | 140 |
| P | phosphorus level | 5 | 53.4 | 145 |
| K | potassium level | 5 | 48.1 | 205 |
| temperature | air temperature, degrees C | 8.8 | 25.6 | 43.7 |
| humidity | relative humidity, % | 14.3 | 71.5 | 100 |
| pH | soil acidity | 3.5 | 6.5 | 9.9 |
| rainfall | rainfall, mm | 20.2 | 103.5 | 298.6 |

**Every stage of the project:**

| Stage / file | Input | Output |
|---|---|---|
| `train.py` | the CSV file | `models/crop_model.onnx`, `crop_model.json`, `crop_mlp.pth`, and `results/report.md`, `confusion_matrix.png`, `metrics.json` |
| `bench.py` | `models/crop_model.onnx` | speed and accuracy numbers, `results/bench.json` |
| `sensor_sim.py` | the CSV file (real rows) | text lines, one per reading, sent over TCP / serial port / screen |
| `live_predict.py` | those text lines | one answer line per reading (or `REJECTED: reason`) |
| `dashboard.py` | those text lines, the dataset file, values typed in the Playground | a web page with Live, Data and Playground tabs (section 8) |

**Text-line format** (what the ESP32 must send, one line per reading, end with a newline):
`N=90,P=42,K=43,T=21.5,H=80.1,PH=6.50,R=202.9` - the keys can come in any order; `R` (rainfall) may be left out.

---

## 3. How the data is read

* **The file**: `datasets/crop_recommendation_kaggle_atharvaingle/Crop_recommendation.csv`, 2200 rows, first lines:
  ```
  N,P,K,temperature,humidity,ph,rainfall,label
  90,42,43,20.87974371,82.00274423,6.502985292000001,202.9355362,rice
  85,58,41,21.77046169,80.31964408,7.038096361,226.6555374,rice
  ```
* **Where it comes from**: Kaggle *Crop Recommendation Dataset* by Atharva Ingle (details, license and checksum:
  `datasets/DATASETS.md`, `datasets/manifest.json`). It is checked by a SHA-256 fingerprint in `selftest.py` so nobody trains on a damaged copy.
* **How it is loaded** - `load_data()` in `src/common.py`:
  1. `pandas` reads the CSV.
  2. The 7 columns become a table `X` of 2200 x 7 numbers (decimal numbers of 32 bits).
  3. The crop names are sorted alphabetically and numbered 0..21 -> the answer list `y`.
* **What the data looks like**: 22 crops x exactly 100 rows each, no missing values, no duplicate rows. Inside one crop the
  7 numbers are almost unrelated to each other (average correlation 0.08), so the data looks computer-generated and clean.
  Real fields are messier, so expect real-world results to be lower than the scores below.
* **The exam split** - `split_data()`: 80 % / 20 % *per crop* (80 + 20 rows of each crop), always the same random choice
  (seed 42). So **1760 rows are for learning and tuning, 440 rows are locked away** and only used to report the final scores.
  Every script uses this same split, so numbers from different scripts are comparable.

---

## 4. How it is trained (`src/train.py`, about 1 minute)

**Step 1 - look at the data.** Prints the row count, duplicates, empty values.

**Step 2 - tune the SVM.** One model type (SVM) has two dials (`C`, `gamma`). 12 combinations are tried with 5-fold
cross-validation *on the 1760 training rows only* and the best pair is kept.

**Step 3 - make "noisy" training data.** Real sensors are not exact (a cheap DHT11 can be 2 C off). If a model only ever sees
perfect numbers, it gets confused by real ones. So the program can add realistic sensor error (`add_sensor_noise()` in `common.py`):

| Quantity | "good" sensors | "cheap" sensors | Based on |
|---|---|---|---|
| N, P, K | +-5 % of the value | +-10 % | DFRobot SEN0605 NPK sensor page ("accuracy <= 5 %") |
| temperature | +-0.5 C | +-2 C | Adafruit guide: DHT22 +-0.5 C, DHT11 +-2 C |
| humidity | +-3 % | +-5 % | Adafruit guide: DHT22 2-5 %, DHT11 5 % |
| pH | +-0.10 | +-0.30 | assumption (one 7-in-1 sensor lists +-0.3 pH) |
| rainfall | +-10 % | +-20 % | assumption (there is no rain sensor in the plan) |

The error is random (bell-curve shaped) with the size above, and values are kept physical (no negative amounts, humidity 0-100, pH 0-14).

**Step 4 - train every model two ways.** Six kinds of model are tried: *Naive Bayes, SVM, Random Forest, LightGBM, XGBoost,
a small neural network (MLP)*. Each is trained
* **plain** - on the 1760 original rows, and
* **noise-aware** - on the 1760 rows *plus 4 noisy copies of them* (8800 rows). Each copy has a random amount of sensor error
  (anything from none up to 1.5 x the "cheap" level), so the model learns that the same crop can show slightly different numbers.

**Step 5 - score each one, fairly.**
* *5-fold cross-validation on the training rows*: the 1760 rows are cut into 5 parts; the model learns from 4 and is tested on the 5th,
  five times. The test part is scored once with perfect numbers and once with "cheap" noise added. This is used **to choose**.
* Then the model learns from all 1760 rows and is scored on the **440 exam rows**: with perfect numbers, with "good" noise and with "cheap"
  noise (average of 20 different noisy copies - the same 20 for every model, so the comparison is fair). Also *top-3 accuracy* (is the right
  crop among the 3 best guesses), macro-F1 and log-loss. This is used **only to report**, never to choose.

**Step 6 - choose the model that gets deployed.** Among the models that can be saved as ONNX (Naive Bayes, SVM, Random Forest,
neural net; LightGBM and XGBoost are compared only), pick the one with the best cross-validated accuracy **with cheap noise**
(ties: best accuracy with perfect numbers). Currently this is **Naive Bayes, noise-aware**. The next best are within
measurement noise of it - with 440 exam rows, one wrong answer is 0.23 %.

**Step 7 - one extra experiment: what if there is no rain sensor?** The chosen model type is trained again with only 6 inputs (rainfall removed).

**Step 8 - save.**

| File | What it is |
|---|---|
| `models/crop_model.onnx` | the chosen model, ready to run (3 KB). This is what `live_predict.py` uses. |
| `models/crop_model.json` | crop names, lowest/highest value of each input (used to check readings), the scores, which model it is |
| `models/crop_mlp.pth` | the small PyTorch neural network, saved separately (if you want to use PyTorch) |
| `results/report.md` | the comparison table of all 12 trained models |
| `results/confusion_matrix.png` | which crops get mixed up on the 440 exam rows |
| `results/metrics.json` | all the numbers in machine-readable form |

**What Naive Bayes does, in plain words.** For every crop it learns, for each of the 7 inputs, the typical value and how much it varies
(for example rice rows: rainfall about 236 mm +- 34, humidity 82 % +- 1.4; chickpea rows: rainfall 80 mm, humidity 17 %). For a new
reading it asks, for each crop: "how normal would these 7 numbers be for this crop?", multiplies the answers together (this assumes the 7
numbers are unrelated - which happens to be true in this dataset) and turns the results into probabilities. It is simple, tiny and fast,
and it works well *here* because of how the dataset is built; it is not a general winner.

**Why rice/jute and maize/cotton get mixed up** (the only 4 mistakes on the 440 exam rows): rice and jute are similar except for rainfall
(236 vs 175 mm); maize and cotton are similar except for humidity (65 vs 80 %). That is also why removing the rainfall input costs about 5 points.

---

## 5. How the saved model is run

The model file is **ONNX**, a neutral format that any computer runs through the `onnxruntime` library - on a CPU, on Windows, Linux or macOS, no PyTorch or scikit-learn needed to *use* it.

| | Name | Type and shape |
|---|---|---|
| input | `X` | decimal numbers, shape `[how many readings, 7]` |
| output 1 | `label` | whole numbers, shape `[how many readings]` - index of the best crop |
| output 2 | `probabilities` | decimals, shape `[how many readings, 22]` - one probability per crop |

Use it in your own code (this exact snippet was run; it prints `rice [('rice', 0.75), ('jute', 0.25), ('coffee', 0.0)]`):

```python
import json, numpy as np, onnxruntime as ort

meta = json.load(open("models/crop_model.json"))
sess = ort.InferenceSession("models/crop_model.onnx", providers=["CPUExecutionProvider"])

x = np.array([[90, 42, 43, 20.9, 82.0, 6.5, 202.9]], dtype=np.float32)  # N P K temperature humidity pH rainfall
label, proba = sess.run(None, {"X": x})
best3 = np.argsort(-proba[0])[:3]
print(meta["classes"][label[0]], [(meta["classes"][i], round(float(proba[0][i]), 2)) for i in best3])
```

**About the percentages.** They are *scores*, not literal chances. The first rice row of the dataset gets "rice 75 %, jute 25 %" even though it is
a clear rice example: noise-aware training deliberately makes the model less over-sure. Trust the **ranking** (and the top 3), not the exact percentage.

---

## 6. How it is run live (`src/live_predict.py`)

For every line that arrives:

1. **Read it** - split at the commas, `KEY=number`. Keys: `N P K T H PH R`.
2. **Check it** - a reading is **rejected** (printed, never given to the model) if it cannot be read, a field is missing, a value is not a
   number (`nan`), or a value is outside the plausible range. Plausible = the range seen in training plus a 25 % margin, and never outside physical limits:

   | Input | N | P | K | temperature | humidity | pH | rainfall |
   |---|---|---|---|---|---|---|---|
   | accepted range | 0..175 | 0..180 | 0..255 | 0.1..52.4 | 0..100 | 1.9..11.5 | 0..368 |

   If rainfall `R` is missing, the typical training value (median) is used and the line is marked with a note - the answer is then less certain.
3. **Predict** - ONNX Runtime on the CPU gives the 22 probabilities (about 0.014 ms for the model itself, 0.2-0.3 ms including Python).
4. **Smooth** - the probabilities of the last 5 accepted readings are averaged, so one odd reading does not flip the recommendation.
   (Right after the crop/field changes, the smoothed answer lags for a few readings - that is intended.)
5. **Print**:
   ```
   #1 rice 64% | top3: rice 64%, jute 36%, coffee 0% | smoothed: rice 64% | 0.30 ms
   #13 REJECTED: missing temperature
   #29 REJECTED: N=9999 outside plausible range 0.0..175.0
   summary: 30 lines, 27 used, 3 rejected, 27/27 correct (100%) against the simulator's truth
   ```
   (`truth: ... OK/WRONG` only appears in `demo`, where the built-in fake sensor knows the right answer.)

**Where the lines come from** (`--source ...`):

| Source | Meaning |
|---|---|
| `demo` | built-in fake sensor in the same program (what the launcher runs) |
| `listen://0.0.0.0:5555` | the laptop waits; the ESP32 / simulator connects over Wi-Fi and sends lines (TCP). *This is the planned ESP32 route.* |
| `tcp://host:port` | the laptop connects to a device that serves lines |
| `serial:COM3` / `serial:/dev/ttyUSB0` | a USB serial port (a real ESP32/Arduino) |
| `stdin` | lines typed or piped in |

---

## 7. The simulator (`sim/sensor_sim.py`) - the stand-in for the ESP32

For each reading it: (1) picks a crop (cycles through all 22, 5 readings each, or one crop with `--crop`), (2) takes a **random real row of that crop**
from the dataset, (3) adds sensor error of the chosen quality (`--preset clean|good|cheap`), (4) writes it as the text line, (5) optionally breaks it on purpose
(`--glitch 0.1` = 10 % of lines get a missing field, `nan`, a crazy value or garbage, to show that the checking works), then sends it (`--out tcp://...`,
`serial:...` or `stdout`) at `--rate` readings per second. The real ESP32 only has to send the same kind of line - nothing on the laptop changes.

---

## 8. The dashboard (`src/dashboard.py` + `src/dashboard.html`)

`python run.py dashboard` opens a page in your web browser (`http://127.0.0.1:8050`). It has three tabs.

* **Input**: the same text lines as `live_predict.py`, from the same sources (`--source demo` is the default fake sensor, `listen://0.0.0.0:5555` for the ESP32, `serial:COM3`, ...);
  plus, in the other tabs, the dataset file and the values *you* type. Options: `--preset clean|good|cheap`, `--glitch 0.08`, `--rate 1` (demo only), `--port 8050`, `--host 0.0.0.0`.
* **Output**: a web page that refreshes every second. Nothing is saved to disk and nothing leaves your computer.

**Live tab** - what the sensors say now:

| Part | What it shows |
|---|---|
| Status (top right) | **Live** when a reading arrived in the last 5 seconds, otherwise *Waiting for sensor* or *No data for N s* |
| Recommended crop | the crop with the best **smoothed** score (average of the last 5 readings), with a confidence bar; in demo mode it also says which crop the fake sensor is simulating |
| Best matches | the 5 best crops, one bar each (scores, not exact chances) |
| Sensors (7 tiles) | latest value of N, P, K, temperature, humidity, pH, rainfall; the **dot** shows where it sits in the range seen in training, the **line** is the last 30 readings (hover for exact values); a warning if a value is more than 10 % outside the training range |
| Recent readings | the last 8 readings as a table, each with its own top crop and score |
| Incoming data | **the raw text lines exactly as they arrived**, newest first, each marked accepted or rejected (with the reason) |
| Session | readings used, readings rejected, model time per reading, correct-vs-simulator (demo only) |

**Data tab** - the data the model learned from. The 2,200 rows of `Crop_recommendation.csv` as a table: pick a crop (or all) to see its typical values (average, spread, lowest, highest of each input)
and page through its rows. Each row also shows the **text line** a sensor would send for it. The *Try* button loads that row into the Playground.

**Playground tab** - change the values and watch the answer:
* seven sliders with number boxes (N, P, K, temperature, humidity, pH, rainfall); the shaded part of each slider is the range present in the training data, and a warning appears if you go outside it;
* start from the **latest sensor reading**, from a **random dataset example** (optionally of one crop you choose), or from a row sent over with *Try*; *Reset* returns to that start;
* the right-hand side shows the recommended crop, the best 5, the **list of what you changed** (e.g. `Nitrogen (N): 89 -> 140`), whether the **recommendation changed** (e.g. `Rice (55%) -> Apple (100%)`) and the text line for the current values (with a *Copy* button);
* the Playground never touches the Live tab: it asks the model separately and does not use the smoothing.

How it works: `dashboard.py` starts one background thread that reads lines and passes each one to the **same** `LivePredictor` class that `live_predict.py` uses (so parsing, checking, smoothing and the model are identical),
and keeps the latest results in memory. A tiny web server (Python standard library only - no extra packages) serves `dashboard.html` and a few JSON pages that the page asks for:

| Page | Gives |
|---|---|
| `/api/state` | the live state (asked every second) |
| `/api/dataset?crop=rice&page=0&size=20` | one page of training rows + the crop's summary |
| `/api/sample?crop=rice` | one random real row |
| `/api/predict?N=90&P=42&K=43&T=20.9&H=82&PH=6.5&R=202.9` | the answer for exactly these values (a bad value comes back as `ok: false` with the reason) |

To change the look, edit `dashboard.html` and refresh the browser.

---

## 9. Reading the results

`results/report.md` has one row per trained model:
* **CV acc (train)** - cross-validation accuracy on the 1760 training rows (+- = spread between the 5 parts).
* **TEST clean / good sensors / cheap sensors** - accuracy on the 440 exam rows with perfect numbers / with "good" / with "cheap" sensor error.
* **Top-3** - how often the right crop is among the 3 best guesses. **Macro-F1** - accuracy that treats all crops equally. **Fit time** - seconds to train.

Current result for the deployed model (Naive Bayes, noise-aware): **99.1 %** clean, **98.0 %** good sensors, **94.3 %** cheap sensors, top-3 100 / 100 / 99.9 %;
without the rainfall input 94.3 % (clean) and 89.5 % (cheap). Speed: 0.014 ms per reading.

---

## 10. Where to change what

| I want to... | Edit |
|---|---|
| use my own sensor error sizes | `PRESETS` in `src/common.py` |
| try another dataset | the path/columns in `common.py` (`DATA_FILE`, `FEATURES`, `load_data`), then `python run.py train` |
| add or remove a model type | `factories()` in `src/train.py` |
| change what counts as a plausible reading | `PHYSICAL` and the margin in `parse_line()` in `src/live_predict.py` |
| change the line format the ESP32 sends | `KEYS` and `parse_line()` in `src/live_predict.py` (and `format_line()` in `sim/sensor_sim.py`) |
| change the dashboard's look or panels | `src/dashboard.html` (layout, colours, text); what data it gets: `State` in `src/dashboard.py` |
| use the model in your own program | the snippet in section 5, or `LivePredictor` in `live_predict.py` |

---

## 11. Limits - read before trusting it

* **The dataset is clean and probably computer-generated.** ~99 % here does not mean 99 % on a real field.
* **Sensor scale.** The training numbers for N, P, K have the scale of this dataset. A real NPK sensor reports mg/kg and may be on a **different scale**.
  Before connecting real sensors, compare their readings with the table in section 2; if they are far off, the readings must be rescaled/calibrated or the model retrained on data from your sensors.
* **The sensor-noise results are simulated** from datasheet figures (two of them - pH and rainfall - are assumptions), not measured on real hardware.
* **There is no rain sensor** in the planned hardware; without rainfall the model is about 5 points worse. Use a rain gauge or weather data if you can.
* **Percentages are not exact probabilities** (section 5).
* **It only matches conditions.** It does not know prices, season, crop rotation, or local rules; it was built from Indian data.
* **Small exam set.** 440 rows: differences of about 0.5 % between models are not meaningful.
