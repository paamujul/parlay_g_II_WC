"""Local dev server mimicking Vercel: serves index.html and /api/live.

Usage:  python3 serve.py   ->  http://localhost:8000
"""

import json
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "api"))
import live  # api/live.py


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.split("?")[0] == "/api/live":
            try:
                body, code = json.dumps(live.build_payload()).encode(), 200
            except Exception as exc:
                body, code = json.dumps({"error": str(exc)}).encode(), 502
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
        else:
            super().do_GET()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"Serving dashboard at http://localhost:{port}")
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
