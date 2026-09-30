"""Standard-library HTTP server for KerrRay Desktop.

``GET /`` and ``GET /static/...`` serve the UI; ``POST /api/<endpoint>``
runs :func:`kerrray.web.api.dispatch` with the JSON body as parameters.
The server binds to 127.0.0.1 only, so the engine is never exposed on the
network. Requests run in threads so the UI stays responsive.
"""

from __future__ import annotations

import json
import mimetypes
import socket
import sys
import traceback
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from kerrray.utils.logging import get_logger
from kerrray.web.api import ApiError, dispatch

STATIC_DIR = Path(__file__).resolve().parent / "static"
HOST = "127.0.0.1"
MAX_BODY_BYTES = 1 << 20

logger = get_logger(__name__)


class KerrRayHandler(BaseHTTPRequestHandler):
    """Serves static files and JSON API calls."""

    server_version = "KerrRayDesktop/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: D102 - quiet access log
        logger.debug("%s - %s", self.address_string(), fmt % args)

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        self._send(status, json.dumps(payload, allow_nan=False, default=float).encode("utf-8"), "application/json")

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        path = self.path.split("?", 1)[0]
        rel = "index.html" if path in ("/", "/index.html") else path.removeprefix("/static/")
        target = (STATIC_DIR / rel).resolve()
        if STATIC_DIR not in target.parents and target != STATIC_DIR / "index.html" or not target.is_file():
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
            return
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/javascript",):
            ctype += "; charset=utf-8"
        self._send(HTTPStatus.OK, target.read_bytes(), ctype)

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        if not self.path.startswith("/api/"):
            self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
            return
        name = self.path.removeprefix("/api/").split("?", 1)[0]
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "request too large"})
            return
        try:
            params = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(params, dict):
                raise ApiError("request body must be a JSON object")
            self._json(HTTPStatus.OK, dispatch(name, params))
        except (ApiError, ValueError) as exc:
            self._json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - report engine failures to the UI
            logger.error("endpoint %s failed:\n%s", name, traceback.format_exc())
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"{type(exc).__name__}: {exc}"})


class ExclusiveServer(ThreadingHTTPServer):
    """Threading server that refuses to share its port with another process.

    ``HTTPServer`` enables SO_REUSEADDR, which on Windows lets a second
    process bind a port that is already in use and silently steal or split
    its traffic. This server disables that and, on Windows, asks for
    SO_EXCLUSIVEADDRUSE.
    """

    allow_reuse_address = False
    daemon_threads = True

    def server_bind(self) -> None:
        if sys.platform == "win32" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def bind(port: int, attempts: int = 20) -> ExclusiveServer:
    """Bind ``port`` or, if it is taken, the next free port above it."""
    last: OSError | None = None
    for candidate in range(port, port + attempts):
        try:
            return ExclusiveServer((HOST, candidate), KerrRayHandler)
        except OSError as exc:
            last = exc
            logger.warning("port %d is in use, trying %d", candidate, candidate + 1)
    raise OSError(f"no free port in {port}..{port + attempts - 1}") from last


def serve(port: int = 8741, *, open_browser: bool = True) -> None:
    """Run the desktop server on ``127.0.0.1:port`` (or the next free port) until interrupted."""
    httpd = bind(port)
    port = httpd.server_address[1]
    url = f"http://{HOST}:{port}/"
    logger.info("KerrRay Desktop running at %s (Ctrl+C to stop)", url)
    if open_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
