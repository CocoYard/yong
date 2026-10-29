"""Local server behind the "Estimate projections" button in power_diagram.html.

    python3 assets/estimate_server.py

Each POST /estimate reloads experiments.py, so edits to `estimate_projections`
apply on the next click. Request (y up, normalized [-1, 1]^2):
    {"shape": str, "loops": [[[x, y], ...], ...], "samples": [{"x", "y", "sdf"}, ...],
     "options": {...}}   # passed to estimate_projections as keyword arguments
Response: {"projections": [[x, y] or null, ...], "norm": {...} or null}, one
projection per sample; a row of NaNs becomes null, which tells the page to drop
that sample's projection point. norm is experiments.LAST_NORM (the refine's
normalization, for the page's normalized Length / T).
"""

import importlib.util
import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HOST, PORT = "127.0.0.1", 8765
EXPERIMENTS = Path(__file__).with_name("experiments.py")


def load_experiments():
    """Import experiments.py fresh, picking up any edits."""
    spec = importlib.util.spec_from_file_location("experiments", EXPERIMENTS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def estimate(data):
    xy = np.array([[s["x"], s["y"]] for s in data["samples"]], dtype=float).reshape(-1, 2)
    sdf = np.array([s["sdf"] for s in data["samples"]], dtype=float)
    options = data.get("options", {})
    mod = load_experiments()
    proj = np.asarray(mod.estimate_projections(xy, sdf, data["loops"], **options), dtype=float)
    if proj.shape != xy.shape:
        raise ValueError(f"estimate_projections returned shape {proj.shape}, expected {xy.shape}")
    return [None if np.isnan(q).any() else q.tolist() for q in proj], getattr(mod, "LAST_NORM", None)


class Handler(BaseHTTPRequestHandler):
    def cors(self):
        # The page may be opened from file://, Jekyll or GitHub Pages
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Private-Network", "true")

    def reply(self, code, body, ctype):
        out = body.encode()
        self.send_response(code)
        self.cors()
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_OPTIONS(self):
        self.send_response(204)
        self.cors()
        self.end_headers()

    def do_POST(self):
        if self.path != "/estimate":
            return self.reply(404, "not found", "text/plain")
        try:
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            proj, norm = estimate(data)
            self.reply(200, json.dumps({"projections": proj, "norm": norm}), "application/json")
            print(f"{data.get('shape')}: {len(proj)} samples", flush=True)
        except Exception:
            tb = traceback.format_exc()
            print(tb, flush=True)
            self.reply(500, tb, "text/plain")

    def log_message(self, *args):
        pass  # keep the console for our own one-line summaries and tracebacks


if __name__ == "__main__":
    print(f"Serving estimate_projections from {EXPERIMENTS} on http://{HOST}:{PORT}/estimate", flush=True)
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
