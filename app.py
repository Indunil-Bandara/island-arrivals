"""Local Python inference service and web frontend. Run python app.py."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs
import json
import pickle
import os
from model import ROOT, features, predict_one

with (ROOT / "models/arrivals.pkl").open("rb") as f:
    models = pickle.load(f)
report = json.loads((ROOT / "models/evaluation.json").read_text(encoding="utf-8"))

class App(BaseHTTPRequestHandler):
    def do_GET(self):
        req = urlsplit(self.path)
        if req.path == "/":
            payload, mime = (ROOT / "index.html").read_bytes(), "text/html; charset=utf-8"
        elif req.path == "/api/report":
            payload, mime = json.dumps(report).encode(), "application/json"
        elif req.path == "/api/predict":
            try:
                prior = int(parse_qs(req.query)["prior"][0])
                if not 0 <= prior <= 350000:
                    raise ValueError()
            except (KeyError, ValueError):
                return self.send_error(400, "Use a whole number between 0 and 350000")
            x = features(1, prior, report["two_back"], report["year_back"])
            payload = json.dumps(predict_one(models, x)).encode()
            mime = "application/json"
        else:
            return self.send_error(404)
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8501"))
    with ThreadingHTTPServer((host, port), App) as server:
        print(f"Listening on http://{host}:{port}", flush=True)
        server.serve_forever()
