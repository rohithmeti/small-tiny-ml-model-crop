"""Live dashboard in your web browser: recommended crop, best matches, the 7 sensor values, recent readings.

    python run.py dashboard                                   # demo with a built-in fake sensor
    python run.py dashboard --source listen://0.0.0.0:5555    # wait for the ESP32 / simulator (same sources as `live`)
    python run.py dashboard --source serial:COM3

INPUT : the same text lines as live_predict.py   N=90,P=42,K=43,T=21.5,H=80.1,PH=6.5,R=202.9
OUTPUT: a web page at http://127.0.0.1:8050 that refreshes every second (dashboard.html). Nothing is stored.
        Tabs:  Live (what the sensors say now, incl. the raw text lines)  |  Data (browse the 2200 training rows)
               |  Playground (move sliders for N, P, K ... and watch the recommendation change)
        Open it from a phone on the same Wi-Fi with  --host 0.0.0.0  and  http://<laptop-ip>:8050
"""
import argparse
import collections
import json
import random
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

from common import FEATURES, load_data
from live_predict import PHYSICAL, LivePredictor, line_source
from sensor_sim import format_line   # (live_predict has put the sim/ folder on the path)

PAGE_FILE = Path(__file__).resolve().parent / "dashboard.html"   # read on every request, so edits show up on refresh


class Dataset:
    """The training file, loaded once, for the Data tab and the "random example" button."""

    def __init__(self):
        self.X, self.y, self.classes = load_data()

    def row(self, i):
        return {"row": int(i) + 1, "values": [round(float(v), 4) for v in self.X[i]],
                "crop": self.classes[self.y[i]], "line": format_line(self.X[i])}

    def indices(self, crop):
        return np.where(self.y == self.classes.index(crop))[0] if crop in self.classes else np.arange(len(self.y))

    def page(self, crop, page, size):
        idx = self.indices(crop)
        sub = self.X[idx]
        summary = [{"feature": f, "mean": round(float(sub[:, j].mean()), 2), "std": round(float(sub[:, j].std()), 2),
                    "min": round(float(sub[:, j].min()), 2), "max": round(float(sub[:, j].max()), 2)} for j, f in enumerate(FEATURES)]
        size = max(1, min(size, 100))
        page = max(0, min(page, (len(idx) - 1) // size))
        return {"crop": crop, "total": int(len(idx)), "page": page, "size": size, "crops": self.classes,
                "rows": [self.row(i) for i in idx[page * size:(page + 1) * size]], "summary": summary}

    def sample(self, crop):
        return self.row(random.choice(self.indices(crop)))


class State:
    """Everything the web page shows. The reader thread writes it, the web server reads it."""

    def __init__(self, pred, source):
        self.lock = threading.Lock()
        self.pred = pred
        margin = 0.25 * (pred.hi - pred.lo)   # same "plausible range" rule as parse_line()
        self.d = {"model": pred.meta["model"], "noise_aware": pred.meta["noise_aware"], "source": source,
                  "features": FEATURES, "range_lo": [float(v) for v in pred.lo], "range_hi": [float(v) for v in pred.hi],
                  "limit_lo": [round(float(max(pred.lo[i] - margin[i], PHYSICAL[i][0])), 2) for i in range(7)],
                  "limit_hi": [round(float(min(pred.hi[i] + margin[i], PHYSICAL[i][1])), 2) for i in range(7)],
                  "seen": 0, "used": 0, "rejected": 0, "right": 0, "known": 0, "ms_sum": 0.0,
                  "latest": None, "last_update": None}
        self.history = collections.deque(maxlen=60)      # last 60 accepted readings (7 numbers each)
        self.recent = collections.deque(maxlen=8)        # rows of the "recent readings" table
        self.rejections = collections.deque(maxlen=4)    # why the last bad lines were refused
        self.lines = collections.deque(maxlen=12)        # the raw text lines as they arrived

    def add(self, line, truth):
        r = self.pred.process(line)
        classes = self.pred.classes
        with self.lock:
            self.d["seen"] += 1
            self.d["last_update"] = time.time()
            clock = time.strftime("%H:%M:%S")
            self.lines.appendleft({"n": self.d["seen"], "t": clock, "line": line.strip()[:120], "ok": r["ok"], "why": r.get("problem")})
            if not r["ok"]:
                self.d["rejected"] += 1
                self.rejections.appendleft({"n": self.d["seen"], "t": clock, "why": r["problem"]})
                return
            self.d["used"] += 1
            self.d["ms_sum"] += r["ms"]
            p, sm = r["p"], r["smoothed"]
            order = np.argsort(-sm)[:5]
            raw = int(p.argmax())
            if truth:
                self.d["known"] += 1
                self.d["right"] += classes[raw] == truth
            values = [float(v) for v in r["x"]]
            self.history.append(values)
            self.recent.appendleft({"n": self.d["seen"], "t": clock, "values": values, "crop": classes[raw], "p": float(p[raw])})
            self.d["latest"] = {"crop": classes[int(order[0])], "confidence": float(sm[order[0]]), "values": values,
                                "top": [{"crop": classes[int(i)], "p": float(sm[i])} for i in order],
                                "note": r["note"], "truth": truth}

    def snapshot(self):
        with self.lock:
            out = dict(self.d)
            out.update(history=list(self.history), recent=list(self.recent), rejections=list(self.rejections), lines=list(self.lines),
                       age=None if out["last_update"] is None else time.time() - out["last_update"])
            return out


class Handler(BaseHTTPRequestHandler):
    state = None
    data = None

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                first = json.dumps(self.state.snapshot()).replace("</", "<\\/").encode()   # current state, so the page is filled at once
                self._send(PAGE_FILE.read_bytes().replace(b"/*INITIAL_STATE*/null", first), "text/html; charset=utf-8")
            elif u.path == "/api/state":
                self._json(self.state.snapshot())
            elif u.path == "/api/dataset":      # the training rows, one page at a time (optionally one crop)
                self._json(self.data.page(q.get("crop", [""])[0], int(q.get("page", ["0"])[0]), int(q.get("size", ["20"])[0])))
            elif u.path == "/api/sample":       # one real row, random (optionally of one crop)
                self._json(self.data.sample(q.get("crop", [""])[0]))
            elif u.path == "/api/predict":      # "what if": answer for the values in the address, without touching the live smoothing
                self._json(self._predict(q))
            else:
                self._send(b"not found", "text/plain", 404)
        except ValueError as e:
            self._send(str(e).encode(), "text/plain", 400)

    def _predict(self, q):
        line = ",".join(f"{k}={q[k][0]}" for k in ("N", "P", "K", "T", "H", "PH", "R") if k in q)
        r = self.state.pred.process(line, remember=False)
        if not r["ok"]:
            return {"ok": False, "problem": r["problem"]}
        order = np.argsort(-r["p"])[:5]
        classes = self.state.pred.classes
        return {"ok": True, "line": format_line(r["x"]), "values": [float(v) for v in r["x"]], "note": r["note"], "ms": r["ms"],
                "top": [{"crop": classes[int(i)], "p": float(r["p"][i])} for i in order]}

    def _json(self, obj):
        self._send(json.dumps(obj).encode(), "application/json")

    def _send(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep the terminal quiet
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="demo")
    ap.add_argument("--preset", default="good", choices=["clean", "good", "cheap"], help="demo source only: sensor quality")
    ap.add_argument("--glitch", type=float, default=0.08, help="demo source only: fraction of broken readings")
    ap.add_argument("--rate", type=float, default=1.0, help="demo source only: readings per second")
    ap.add_argument("--smooth", type=int, default=5, help="average the probabilities of the last N good readings")
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to open the page from another device")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--no-browser", action="store_true")
    a = ap.parse_args()

    pred = LivePredictor(a.smooth)
    state = State(pred, a.source)
    Handler.state = state
    Handler.data = Dataset()
    try:
        server = ThreadingHTTPServer((a.host, a.port), Handler)
    except OSError as e:
        raise SystemExit(f"cannot start on {a.host}:{a.port} ({e}). Is the dashboard already running? Try --port {a.port + 1}")
    server.daemon_threads = True

    def reader():
        for line, truth in line_source(a.source, a.preset, a.glitch, delay=1.0 / a.rate, hold=15):
            state.add(line, truth)

    threading.Thread(target=reader, daemon=True).start()
    url = f"http://{'127.0.0.1' if a.host == '0.0.0.0' else a.host}:{a.port}"
    print(f"model: {pred.meta['model']} (noise-aware={pred.meta['noise_aware']})   source: {a.source}")
    print(f"Dashboard: {url}   (Ctrl+C to stop)", flush=True)
    if a.host == "0.0.0.0":
        print(f"From another device on the same network open  http://<this-laptop-ip>:{a.port}", flush=True)
    if not a.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
