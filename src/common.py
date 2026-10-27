"""Shared helpers used by every script: paths, data loading, sensor-noise model, small neural net.

Nothing here runs on its own; the other scripts import it.
"""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "datasets/crop_recommendation_kaggle_atharvaingle/Crop_recommendation.csv"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]  # the 7 model inputs, in this order
SEED = 42


# ----------------------------------------------------------------------------- data
def load_data():
    """-> X (2200 x 7 float32), y (class index 0..21), classes (list of crop names)."""
    import pandas as pd
    if not DATA_FILE.exists():
        raise SystemExit("Dataset missing. Run:  python run.py data")
    df = pd.read_csv(DATA_FILE)
    classes = sorted(df["label"].unique())
    y = df["label"].map({c: i for i, c in enumerate(classes)}).to_numpy()
    return df[FEATURES].to_numpy(np.float32), y, classes


def split_data(X, y):
    """The one fixed 80/20 split (1760 train / 440 test) used everywhere, so results are comparable."""
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


# ----------------------------------------------------------------------------- where things live
def get_device():
    """This package is CPU-only."""
    return "cpu"


def out_dir(kind):
    """models/ or results/ (created on demand)."""
    d = ROOT / kind
    d.mkdir(parents=True, exist_ok=True)
    return d


def find_model_dir():
    d = ROOT / "models"
    if not (d / "crop_model.onnx").exists():
        raise SystemExit("No trained model found. Run:  python run.py train")
    return d


# ----------------------------------------------------------------------------- sensor noise
# Real sensors are not perfect. sigma values below are 'one standard deviation' of the error.
#   good : DHT22 (+-0.5 C, 2-5 % RH)  + NPK sensor 'within 5 %' (DFRobot SEN0605)  + decent pH probe
#   cheap: DHT11 (+-2 C, +-5 % RH)    + budget NPK/pH (7-in-1 sensors quote +-0.3 pH)
# The pH 0.1 figure and the rainfall figures are ASSUMPTIONS (no rain sensor is in the planned hardware).
PRESETS = {
    "clean": None,
    "good": dict(npk=0.05, temp=0.5, hum=3.0, ph=0.10, rain=0.10),
    "cheap": dict(npk=0.10, temp=2.0, hum=5.0, ph=0.30, rain=0.20),
}


def add_sensor_noise(X, preset, rng, scale=1.0):
    """Return a copy of X (n x 7) with realistic sensor error added (N,P,K,rain: relative; others: absolute)."""
    p = PRESETS[preset]
    Y = np.array(X, dtype=np.float64)
    if p is None or scale == 0:
        return Y.astype(np.float32)
    n = len(Y)
    for i in range(3):
        Y[:, i] *= 1 + rng.normal(0, p["npk"] * scale, n)
    Y[:, 3] += rng.normal(0, p["temp"] * scale, n)
    Y[:, 4] += rng.normal(0, p["hum"] * scale, n)
    Y[:, 5] += rng.normal(0, p["ph"] * scale, n)
    Y[:, 6] *= 1 + rng.normal(0, p["rain"] * scale, n)
    Y[:, [0, 1, 2, 6]] = np.maximum(Y[:, [0, 1, 2, 6]], 0)
    Y[:, 4] = np.clip(Y[:, 4], 0, 100)
    Y[:, 5] = np.clip(Y[:, 5], 0, 14)
    return Y.astype(np.float32)


# ----------------------------------------------------------------------------- small neural network
class TorchMLP:
    """7 inputs -> 64 -> 64 -> 22 classes. Standardises inputs itself. Runs on the CPU."""

    def __init__(self, hidden=(64, 64), epochs=120, lr=3e-3, batch=256, wd=1e-4, device="cpu", seed=SEED):
        self.hidden, self.epochs, self.lr, self.batch, self.wd = hidden, epochs, lr, batch, wd
        self.device, self.seed = device, seed

    def _build(self, n_in, n_out):
        import torch.nn as nn
        layers, d = [], n_in
        for h in self.hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        return nn.Sequential(*layers, nn.Linear(d, n_out))

    def fit(self, X, y):
        import torch
        import torch.nn.functional as F
        torch.manual_seed(self.seed)
        self.mean, self.std = X.mean(0).astype(np.float32), (X.std(0) + 1e-8).astype(np.float32)
        self.n_classes = int(y.max()) + 1
        self.net = self._build(X.shape[1], self.n_classes).to(self.device)
        Xt = torch.tensor((X - self.mean) / self.std, dtype=torch.float32, device=self.device)
        yt = torch.tensor(y, dtype=torch.long, device=self.device)
        n = len(Xt)
        steps = self.epochs * ((n + self.batch - 1) // self.batch)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        gen = torch.Generator().manual_seed(self.seed)  # fixed shuffling order -> repeatable training
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=gen).to(self.device)
            for i in range(0, n, self.batch):
                idx = perm[i:i + self.batch]
                loss = F.cross_entropy(self.net(Xt[idx]), yt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
        self.net.eval()
        self.classes_ = np.arange(self.n_classes)
        return self

    def predict_proba(self, X):
        import torch
        with torch.no_grad():
            Xt = torch.tensor((X - self.mean) / self.std, dtype=torch.float32, device=self.device)
            return torch.softmax(self.net(Xt), dim=1).cpu().numpy()

    def predict(self, X):
        return self.predict_proba(X).argmax(1)

    def layers(self):
        """[(weight (out x in), bias)] as numpy arrays, for ONNX export."""
        return [(m.weight.detach().cpu().numpy(), m.bias.detach().cpu().numpy())
                for m in self.net if hasattr(m, "weight")]

    def save_pth(self, path, classes):
        import torch
        torch.save({"state_dict": {k: v.cpu() for k, v in self.net.state_dict().items()},
                    "mean": self.mean, "std": self.std, "hidden": self.hidden,
                    "classes": classes, "features": FEATURES}, path)

    @classmethod
    def load_pth(cls, path, device="cpu"):
        """Rebuild a network saved by save_pth (our own file, so weights_only=False is fine)."""
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = cls(hidden=tuple(ck["hidden"]), device=device)
        m.mean, m.std, m.n_classes = ck["mean"], ck["std"], len(ck["classes"])
        m.net = m._build(len(m.mean), m.n_classes)
        m.net.load_state_dict(ck["state_dict"])
        m.net.to(device).eval()
        m.classes_ = np.arange(m.n_classes)
        return m
