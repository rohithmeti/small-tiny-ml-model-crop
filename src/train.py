"""Train several models on the crop data, compare them fairly, pick one, save it as ONNX.

    python run.py train            # full run (about 1 minute on a laptop CPU)
    python run.py train --quick    # fast smoke test

INPUT : datasets/crop_recommendation_kaggle_atharvaingle/Crop_recommendation.csv  (7 numbers -> 1 of 22 crops)
OUTPUT: models/crop_model.onnx   the deployed model (raw 7 numbers in -> crop probabilities out)
        models/crop_model.json   crop names, input ranges, scores
        models/crop_mlp.pth      the small PyTorch network
        results/report.md        comparison table;  metrics.json;  confusion_matrix.png

Method (kept deliberately simple):
  1. One fixed split: 1760 rows to train/tune, 440 rows locked away as the final exam.
  2. Every model is trained twice: normally, and with 'noise-aware' training (extra copies of the training rows
     with realistic sensor error added) so it copes with imperfect sensors.
  3. Models are chosen with 5-fold cross-validation on the 1760 training rows only; the 440 test rows are only
     used to report the final scores.
"""
import argparse
import json
import time
import warnings

import numpy as np

from common import (FEATURES, SEED, TorchMLP, add_sensor_noise, get_device, load_data, out_dir, split_data)

warnings.filterwarnings("ignore")
ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true", help="tiny/fast run for testing the setup")
args = ap.parse_args()

from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import f1_score, log_loss  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

DEVICE = get_device()
RES, MOD = out_dir("results"), out_dir("models")
CV_FOLDS = 3 if args.quick else 5
DEPLOYABLE = ["NaiveBayes", "SVM", "RandomForest", "MLP"]  # can be exported to ONNX (LightGBM/XGBoost: comparison only)
print("compute device: CPU")

# ------------------------------------------------------------------ 1. data + quick sanity facts
X, y, classes = load_data()
Xtr, Xte, ytr, yte = split_data(X, y)
print(f"data: {len(X)} rows, {len(classes)} crops, train {len(Xtr)} / test {len(Xte)}, "
      f"duplicates {len(X) - len(np.unique(X, axis=0))}, NaN {int(np.isnan(X).sum())}")
corr = np.mean([np.abs(np.corrcoef(X[y == c], rowvar=False)[np.triu_indices(7, 1)]).mean() for c in range(len(classes))])
print(f"inside one crop the 7 features are almost uncorrelated (mean |corr| = {corr:.2f}) -> "
      "easy, 'synthetic-like' data: expect very high accuracy")


# ------------------------------------------------------------------ 2. models
def tune_svm():
    gs = GridSearchCV(make_pipeline(StandardScaler(), SVC()),
                      {"svc__C": [10, 100, 1000], "svc__gamma": [0.02, 0.05, 0.1, "scale"]}, cv=CV_FOLDS, n_jobs=-1)
    gs.fit(Xtr, ytr)
    print(f"SVM tuned on training rows: {gs.best_params_}  (cv acc {gs.best_score_:.4f})")
    return dict(C=gs.best_params_["svc__C"], gamma=gs.best_params_["svc__gamma"])


svm_params = tune_svm()


def factories(prob=False):
    f = {
        "NaiveBayes": lambda: GaussianNB(),
        "SVM": lambda: make_pipeline(StandardScaler(), SVC(probability=prob, random_state=SEED, **svm_params)),
        "RandomForest": lambda: RandomForestClassifier(300, n_jobs=-1, random_state=SEED),
        "LightGBM": lambda: LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                                           random_state=SEED, verbose=-1, n_jobs=4),
        "XGBoost": lambda: XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4, tree_method="hist",
                                         device=DEVICE, random_state=SEED, n_jobs=4),
        "MLP": lambda: TorchMLP(epochs=30 if args.quick else 120, device=DEVICE),
    }
    return {k: v for k, v in f.items() if not (args.quick and k in ("LightGBM", "XGBoost"))}


def scores(m, X_):
    return m.predict_proba(X_) if hasattr(m, "predict_proba") else m.decision_function(X_)


def augment(X_, y_, rng, copies=4):
    """Noise-aware training data: the original rows + copies with random sensor error (0 .. 1.5 x 'cheap')."""
    Xs, ys = [X_], [y_]
    for _ in range(copies):
        Xs.append(add_sensor_noise(X_, "cheap", rng, scale=rng.uniform(0, 1.5)))
        ys.append(y_)
    return np.vstack(Xs), np.concatenate(ys)


def acc(m, X_, y_):
    return float((m.predict(X_) == y_).mean())


# ------------------------------------------------------------------ 3. choose with cross-validation (train rows only)
def cv_eval(factory, aug):
    clean, noisy = [], []
    for k, (a, b) in enumerate(StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED).split(Xtr, ytr)):
        Xa, ya = augment(Xtr[a], ytr[a], np.random.default_rng(SEED + k)) if aug else (Xtr[a], ytr[a])
        m = factory().fit(Xa, ya)
        clean.append(acc(m, Xtr[b], ytr[b]))
        noisy.append(acc(m, add_sensor_noise(Xtr[b], "cheap", np.random.default_rng(1000 + k)), ytr[b]))
    return float(np.mean(clean)), float(np.std(clean)), float(np.mean(noisy))


# the same 20 noisy copies of the test set are given to every model (fair comparison)
NOISY_TEST = {p: [add_sensor_noise(Xte, p, np.random.default_rng(5000 + d)) for d in range(20)] for p in ("good", "cheap")}


def final_eval(m):
    s = scores(m, Xte)
    top3 = float(np.mean([yte[i] in np.argsort(-s[i])[:3] for i in range(len(yte))]))
    r = {"test_acc": acc(m, Xte, yte), "macro_f1": float(f1_score(yte, m.predict(Xte), average="macro")), "top3": top3}
    if hasattr(m, "predict_proba"):
        r["log_loss"] = float(log_loss(yte, m.predict_proba(Xte), labels=np.arange(len(classes))))
    for p, copies in NOISY_TEST.items():
        r[f"acc_{p}_noise"] = float(np.mean([acc(m, c, yte) for c in copies]))
    return r


rows, fitted = [], {}
t_all = time.time()
for name, make in factories().items():
    for aug in (False, True):
        t0 = time.time()
        cv_mean, cv_std, cv_noisy = cv_eval(make, aug)
        Xa, ya = augment(Xtr, ytr, np.random.default_rng(SEED)) if aug else (Xtr, ytr)
        t1 = time.time()
        m = make().fit(Xa, ya)
        fit_s = time.time() - t1
        r = dict(model=name, noise_aware=aug, cv_acc=cv_mean, cv_std=cv_std, cv_acc_noisy=cv_noisy, fit_seconds=fit_s,
                 **final_eval(m))
        rows.append(r)
        fitted[(name, aug)] = m
        print(f"{name:<11} {'noise-aware' if aug else 'plain      '} cv {cv_mean:.4f}+-{cv_std:.4f} | cv@noise {cv_noisy:.4f} | "
              f"TEST clean {r['test_acc']:.4f} good {r['acc_good_noise']:.4f} cheap {r['acc_cheap_noise']:.4f} | "
              f"fit {fit_s:.1f}s ({time.time() - t0:.0f}s incl. cv)")
print(f"all comparisons took {time.time() - t_all:.0f}s")

# ------------------------------------------------------------------ 4. pick the deployed model (by cross-validation only)
cand = [r for r in rows if r["model"] in DEPLOYABLE]
best = max(cand, key=lambda r: (round(r["cv_acc_noisy"], 4), r["cv_acc"]))
print(f"\nDEPLOYED: {best['model']} ({'noise-aware' if best['noise_aware'] else 'plain'}) "
      f"- best cross-validated accuracy on noisy sensor data ({best['cv_acc_noisy']:.4f})")

if best["model"] == "SVM":  # the deployed SVM needs probabilities (confidence) -> refit with probability=True
    Xa, ya = augment(Xtr, ytr, np.random.default_rng(SEED)) if best["noise_aware"] else (Xtr, ytr)
    final = factories(prob=True)["SVM"]().fit(Xa, ya)
else:
    final = fitted[(best["model"], best["noise_aware"])]

# ------------------------------------------------------------------ 5. extra: what if there is NO rain sensor?
Xa, ya = augment(Xtr, ytr, np.random.default_rng(SEED)) if best["noise_aware"] else (Xtr, ytr)
m6 = factories(prob=False)[best["model"]]().fit(Xa[:, :6], ya)
no_rain = {"clean": acc(m6, Xte[:, :6], yte)}
for p, copies in NOISY_TEST.items():
    no_rain[p] = float(np.mean([acc(m6, c[:, :6], yte) for c in copies]))
print(f"without the rainfall input (6 features): clean {no_rain['clean']:.4f}  good {no_rain['good']:.4f}  cheap {no_rain['cheap']:.4f}")

# ------------------------------------------------------------------ 6. save: ONNX + json + pth + report
import onnx  # noqa: E402
from onnx import TensorProto, helper, numpy_helper  # noqa: E402
from skl2onnx import to_onnx  # noqa: E402


def mlp_to_onnx(mlp, path):
    """Write the MLP as a small ONNX graph by hand: (x-mean)/std -> Linear+ReLU ... -> Softmax. Outputs [label, probabilities]."""
    inits = [numpy_helper.from_array(mlp.mean, "mean"), numpy_helper.from_array(mlp.std, "std")]
    nodes = [helper.make_node("Sub", ["input", "mean"], ["centered"]), helper.make_node("Div", ["centered", "std"], ["h0"])]
    ls = mlp.layers()
    for i, (W, b) in enumerate(ls):
        inits += [numpy_helper.from_array(W.astype(np.float32), f"W{i}"), numpy_helper.from_array(b.astype(np.float32), f"b{i}")]
        last = i == len(ls) - 1
        nodes.append(helper.make_node("Gemm", [f"h{i}", f"W{i}", f"b{i}"], ["logits" if last else f"z{i}"], transB=1))
        if not last:
            nodes.append(helper.make_node("Relu", [f"z{i}"], [f"h{i + 1}"]))
    nodes += [helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
              helper.make_node("ArgMax", ["probabilities"], ["label"], axis=1, keepdims=0)]
    g = helper.make_graph(nodes, "crop_mlp", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None, 7])],
                          [helper.make_tensor_value_info("label", TensorProto.INT64, [None]),
                           helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, len(classes)])], inits)
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


onnx_path = MOD / "crop_model.onnx"
if best["model"] == "MLP":
    mlp_to_onnx(final, onnx_path)
else:
    onx = to_onnx(final, Xtr[:1], options={id(final): {"zipmap": False}}, target_opset=15)
    onnx_path.write_bytes(onx.SerializeToString())

import onnxruntime as ort  # noqa: E402

sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
agree = float((sess.run(None, {sess.get_inputs()[0].name: Xte})[0] == final.predict(Xte)).mean())
print(f"ONNX file {onnx_path.stat().st_size / 1024:.0f} KB; ONNX Runtime (CPU) agrees with the trained model on {agree:.2%} of test rows")

# the PyTorch network is always saved as .pth too (best MLP variant by cross-validated noisy accuracy)
mlp_rows = [r for r in rows if r["model"] == "MLP"]
mlp_best = max(mlp_rows, key=lambda r: (round(r["cv_acc_noisy"], 4), r["cv_acc"]))
fitted[("MLP", mlp_best["noise_aware"])].save_pth(MOD / "crop_mlp.pth", classes)

lo, hi = X.min(0), X.max(0)
(MOD / "crop_model.json").write_text(json.dumps({
    "model": best["model"], "noise_aware": best["noise_aware"], "device_used_for_training": DEVICE,
    "features": FEATURES, "classes": classes, "onnx_input_name": sess.get_inputs()[0].name,
    "input_min": [float(v) for v in lo], "input_max": [float(v) for v in hi],
    "scores": best, "scores_without_rainfall": no_rain, "onnx_agreement": agree}, indent=2))
(RES / "metrics.json").write_text(json.dumps({"rows": rows, "deployed": best, "svm_params": {k: str(v) for k, v in svm_params.items()},
                                              "without_rainfall": no_rain, "device": DEVICE}, indent=2))

lines = [f"# Training report ({DEVICE.upper()} run)", "",
         f"Test set = {len(yte)} rows never used for training or model choice. 'good'/'cheap' = sensor noise added to the test rows "
         "(mean of 20 noisy copies; see src/common.py).", "",
         "| Model | Noise-aware training | CV acc (train) | TEST clean | TEST good sensors | TEST cheap sensors | Top-3 | Macro-F1 | Fit time (s) |",
         "|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    mark = " **<- deployed**" if (r["model"], r["noise_aware"]) == (best["model"], best["noise_aware"]) else ""
    lines.append(f"| {r['model']}{mark} | {'yes' if r['noise_aware'] else 'no'} | {r['cv_acc']:.3f} +- {r['cv_std']:.3f} | "
                 f"{r['test_acc']:.3f} | {r['acc_good_noise']:.3f} | {r['acc_cheap_noise']:.3f} | {r['top3']:.3f} | "
                 f"{r['macro_f1']:.3f} | {r['fit_seconds']:.1f} |")
lines += ["", f"Without the rainfall input ({best['model']}): clean {no_rain['clean']:.3f}, good {no_rain['good']:.3f}, cheap {no_rain['cheap']:.3f}.", ""]
(RES / "report.md").write_text("\n".join(lines))

try:  # confusion matrix of the deployed model on the clean test rows
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(10, 9))
    ConfusionMatrixDisplay.from_predictions(yte, final.predict(Xte), display_labels=classes, xticks_rotation=90, ax=ax, colorbar=False)
    ax.set_title(f"{best['model']} - confusion matrix on the 440 test rows")
    fig.tight_layout()
    fig.savefig(RES / "confusion_matrix.png", dpi=110)
except Exception as e:  # noqa: BLE001
    print("(confusion matrix not drawn:", e, ")")
print(f"saved: {MOD}/crop_model.onnx|json|pth and {RES}/report.md")
