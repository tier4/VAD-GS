#!/usr/bin/env python3
"""Simple HTTP server for the Cesium 3D Tiles viewer.

Serves both the viewer HTML and the tileset files with correct CORS headers.

Usage:
    python viewer/serve.py [--port 8080] [--tiles path/to/tiles]

    Then open http://localhost:8080/?tileset=tiles/tileset.json
"""

from __future__ import annotations

import argparse
import functools
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


class CORSHandler(SimpleHTTPRequestHandler):
    """HTTP handler with CORS headers and correct MIME types for 3D Tiles."""

    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".glb": "model/gltf-binary",
        ".gltf": "model/gltf+json",
        ".json": "application/json",
        ".js": "application/javascript",
        ".wasm": "application/wasm",
    }

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        # Compact logging
        sys.stderr.write(f"  {args[0]}\n")


def main():
    parser = argparse.ArgumentParser(description="Serve Cesium 3D Tiles viewer")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument(
        "--tiles",
        type=Path,
        default=None,
        help="Path to tiles directory. A symlink 'tiles' will be created in viewer/",
    )
    args = parser.parse_args()

    viewer_dir = Path(__file__).parent.resolve()
    os.chdir(viewer_dir)

    # Create symlink to tiles directory if specified
    if args.tiles:
        tiles_path = args.tiles.resolve()
        link_path = viewer_dir / "tiles"
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink()
        link_path.symlink_to(tiles_path)
        tileset_url = "tiles/tileset.json"
        print(f"Tiles linked: {tiles_path} -> {link_path}")
    else:
        tileset_url = "tileset.json"

    handler = CORSHandler
    httpd = HTTPServer(("0.0.0.0", args.port), handler)

    url = f"http://localhost:{args.port}/?tileset={tileset_url}"
    print(f"Serving viewer at: {url}")
    print("Press Ctrl+C to stop.")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        httpd.server_close()


if __name__ == "__main__":
    main()
