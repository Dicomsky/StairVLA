"""HTTPS + secure-WebSocket server for the WebXR teleoperation page.

One port serves both the page (``web/``) and the ``wss://`` controller stream, so the
headset only has to accept the self-signed certificate once. The certificate is generated
locally on first run (``openssl``) and never shipped with the repository.

Headset -> server: ``{"type": "controllers", "left": {...}, "right": {...}}`` every frame.
Server -> headset: ``{"type": "status", ...}`` (shown on the in-headset panel) and
``{"type": "haptic", "hand": "right", "intensity": 0.5, "duration_ms": 60}``.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import socket
import ssl
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

WEB_DIR = Path(__file__).resolve().parent / "web"
mimetypes.add_type("model/gltf-binary", ".glb")
mimetypes.add_type("model/gltf+json", ".gltf")
DEFAULT_TLS_DIR = Path.home() / ".cache" / "stairvla" / "vr_tls"


@dataclass
class ControllerState:
    position: np.ndarray  # WebXR local-floor frame, metres
    quaternion: np.ndarray | None  # [x, y, z, w]
    trigger: float = 0.0
    squeeze: bool = False
    buttons: dict[str, bool] = field(default_factory=dict)
    received_t: float = 0.0

    def pressed(self, name: str) -> bool:
        return bool(self.buttons.get(name, False))


def _parse_controller(data: dict[str, Any] | None, received_t: float) -> ControllerState | None:
    if not data:
        return None
    pos = data.get("position") or {}
    if not all(k in pos for k in ("x", "y", "z")):
        return None
    position = np.array([pos["x"], pos["y"], pos["z"]], dtype=float)
    if not np.all(np.isfinite(position)) or np.linalg.norm(position) < 1e-4:
        return None  # controller not tracked yet
    quat = data.get("quaternion") or {}
    quaternion = (
        np.array([quat["x"], quat["y"], quat["z"], quat["w"]], dtype=float)
        if all(k in quat for k in ("x", "y", "z", "w"))
        else None
    )
    buttons = {str(k): bool(v) for k, v in (data.get("buttons") or {}).items()}
    return ControllerState(
        position=position,
        quaternion=quaternion,
        trigger=float(data.get("trigger", 0.0) or 0.0),
        squeeze=bool(buttons.get("squeeze", False) or data.get("gripActive", False)),
        buttons=buttons,
        received_t=received_t,
    )


def local_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def ensure_tls_certificate(cert: Path | None, key: Path | None) -> tuple[Path, Path]:
    """Return (cert, key); generate a self-signed pair in ~/.cache/stairvla/vr_tls if none is given."""
    if cert is not None and key is not None:
        return Path(cert), Path(key)
    cert_path = DEFAULT_TLS_DIR / "cert.pem"
    key_path = DEFAULT_TLS_DIR / "key.pem"
    if cert_path.exists() and key_path.exists():
        return cert_path, key_path
    DEFAULT_TLS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[VR] Generating a self-signed TLS certificate in {DEFAULT_TLS_DIR} ...")
    try:
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-sha256", "-days", "3650",
                "-keyout", str(key_path), "-out", str(cert_path), "-subj", "/CN=stairvla-vr-teleop",
            ],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("openssl is required to create the VR TLS certificate (apt install openssl).") from exc
    key_path.chmod(0o600)
    return cert_path, key_path


class VRServer:
    """Runs in a background thread; the robot loop reads :meth:`latest` and pushes :meth:`publish`."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8443, cert: Path | None = None, key: Path | None = None):
        self.host = host
        self.port = port
        self.cert, self.key = ensure_tls_certificate(cert, key)
        self._lock = threading.Lock()
        self._controllers: dict[str, ControllerState | None] = {"left": None, "right": None}
        self._last_packet_t = 0.0
        self._clients: set[Any] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._stop: asyncio.Event | None = None
        self._error: BaseException | None = None
        self._status: dict[str, Any] = {}

    @property
    def url(self) -> str:
        host = local_ip() if self.host in ("0.0.0.0", "") else self.host
        return f"https://{host}:{self.port}"

    # ------------------------------------------------------------------ lifecycle
    def start(self, timeout_s: float = 10.0) -> None:
        self._thread = threading.Thread(target=self._run, name="vr-server", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout_s):
            raise RuntimeError("VR server did not start in time.")
        if self._error is not None:
            if isinstance(self._error, OSError) and self._error.errno == 98:
                raise RuntimeError(
                    f"Port {self.port} is already in use (is another record.py running?). Stop it or pass --vr-port."
                ) from self._error
            raise RuntimeError(f"VR server failed to start: {self._error}") from self._error

    def stop(self) -> None:
        if self._loop is not None and self._stop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._stop.set)
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _run(self) -> None:
        try:
            asyncio.run(self._serve())
        except BaseException as exc:  # noqa: BLE001 - surfaced to the caller of start()
            self._error = exc
            self._ready.set()

    async def _serve(self) -> None:
        from websockets.asyncio.server import serve

        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(certfile=str(self.cert), keyfile=str(self.key))
        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        async with serve(
            self._handle_ws,
            self.host,
            self.port,
            ssl=ssl_context,
            process_request=self._process_http,
            max_size=2**20,
            ping_interval=None,
        ):
            self._ready.set()
            await self._stop.wait()

    # ------------------------------------------------------------------ http
    def _process_http(self, connection, request):
        """Serve the static page for plain HTTPS requests; let WebSocket upgrades through."""
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return None
        from websockets.datastructures import Headers
        from websockets.http11 import Response

        path = request.path.split("?", 1)[0]
        name = "index.html" if path in ("", "/") else path.lstrip("/")
        file_path = (WEB_DIR / name).resolve()
        if WEB_DIR.resolve() not in file_path.parents or not file_path.is_file():
            return connection.respond(404, "Not found\n")
        body = file_path.read_bytes()
        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        headers = Headers([("Content-Type", content_type), ("Content-Length", str(len(body))), ("Cache-Control", "no-store")])
        return Response(200, "OK", headers, body)

    # ------------------------------------------------------------------ websocket
    async def _handle_ws(self, websocket) -> None:
        self._clients.add(websocket)
        print(f"[VR] Headset connected from {websocket.remote_address[0]}.")
        try:
            if self._status:
                await websocket.send(json.dumps({"type": "status", **self._status}))
            async for message in websocket:
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "controllers":
                    now = time.perf_counter()
                    left = _parse_controller(data.get("left"), now)
                    right = _parse_controller(data.get("right"), now)
                    with self._lock:
                        self._controllers = {"left": left, "right": right}
                        self._last_packet_t = now
        except Exception as exc:  # noqa: BLE001 - a dropped headset must not kill the server
            print(f"[VR] Headset connection closed: {exc}")
        finally:
            self._clients.discard(websocket)
            with self._lock:
                self._controllers = {"left": None, "right": None}
            print("[VR] Headset disconnected.")

    def latest(self) -> tuple[ControllerState | None, ControllerState | None, float]:
        """(left, right, age of the newest packet in seconds; inf if none)."""
        with self._lock:
            age = time.perf_counter() - self._last_packet_t if self._last_packet_t else float("inf")
            return self._controllers["left"], self._controllers["right"], age

    @property
    def num_clients(self) -> int:
        return len(self._clients)

    def publish(self, message: dict[str, Any]) -> None:
        """Send a JSON message to every connected headset (thread-safe, non-blocking)."""
        if message.get("type") == "status":
            self._status = {k: v for k, v in message.items() if k != "type"}
        if self._loop is None or not self._clients:
            return
        payload = json.dumps(message)

        async def _send() -> None:
            for client in list(self._clients):
                try:
                    await client.send(payload)
                except Exception:  # noqa: BLE001
                    pass

        asyncio.run_coroutine_threadsafe(_send(), self._loop)

    def haptic(self, hand: str = "right", intensity: float = 0.5, duration_ms: int = 60) -> None:
        self.publish({"type": "haptic", "hand": hand, "intensity": intensity, "duration_ms": duration_ms})
