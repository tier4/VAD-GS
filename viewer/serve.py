#!/usr/bin/env python3
"""Simple HTTP server for the Cesium 3D Tiles viewer.

Serves viewer/index.html at / and tiles files at /tiles/.
No symlinks needed — routes requests to the correct directory.

Usage:
    python viewer/serve.py --tiles output/t4_exp/.../cesium_tiles/iteration_30000 --port 9000
"""

from __future__ import annotations

import argparse
import mimetypes
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# Ensure correct MIME types
mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("model/gltf+json", ".gltf")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("application/wasm", ".wasm")

VIEWER_DIR = Path(__file__).parent.resolve()


def make_handler(tiles_dir: Path):
    """Create a handler class bound to the given tiles directory."""

    class Handler(BaseHTTPRequestHandler):

        def do_GET(self):
            # Strip query string
            path = self.path.split("?")[0]

            if path == "/" or path == "/index.html":
                self._serve_file(VIEWER_DIR / "index.html")
            elif path.startswith("/tiles/"):
                rel = path[len("/tiles/"):]
                self._serve_file(tiles_dir / rel)
            else:
                # Try viewer dir for other assets (css, js, etc.)
                self._serve_file(VIEWER_DIR / path.lstrip("/"))

        def do_OPTIONS(self):
            self.send_response(200)
            self._cors_headers()
            self.end_headers()

        def _serve_file(self, filepath: Path):
            filepath = filepath.resolve()
            if not filepath.is_file():
                self.send_response(404)
                self._cors_headers()
                self.end_headers()
                self.wfile.write(f"404 Not Found: {self.path}\n".encode())
                return
            content = filepath.read_bytes()
            mime, _ = mimetypes.guess_type(str(filepath))
            self.send_response(200)
            self.send_header("Content-Type", mime or "application/octet-stream")
            self.send_header("Content-Length", str(len(content)))
            self._cors_headers()
            self.end_headers()
            self.wfile.write(content)

        def _cors_headers(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")

        def log_message(self, format, *args):
            sys.stderr.write(f"  {args[0]}\n")

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Serve Cesium 3D Tiles viewer")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--tiles", type=Path, required=True, help="Path to tiles directory")
    args = parser.parse_args()

    tiles_dir = args.tiles.resolve()
    if not (tiles_dir / "tileset.json").is_file():
        print(f"Error: tileset.json not found in {tiles_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Tiles dir: {tiles_dir}")
    print(f"  /        -> {VIEWER_DIR / 'index.html'}")
    print(f"  /tiles/* -> {tiles_dir}/")

    httpd = HTTPServer(("0.0.0.0", args.port), make_handler(tiles_dir))

    print(f"\nhttp://localhost:{args.port}/")
    print("Press Ctrl+C to stop.")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        httpd.server_close()


if __name__ == "__main__":
    main()
