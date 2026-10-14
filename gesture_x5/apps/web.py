#!/usr/bin/env python3
"""Small dependency-free web viewer for the RDK X5 USB camera."""

from __future__ import annotations

import argparse
import json
import signal
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2


INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>RDK X5 Camera</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #0b0e14; color: #e8edf5; }
    header { display: flex; gap: 12px; align-items: center; flex-wrap: wrap;
             padding: 14px 18px; background: #141a24; }
    h1 { font-size: 18px; margin: 0 14px 0 0; }
    button { border: 1px solid #39465c; background: #202938; color: inherit;
             border-radius: 7px; padding: 8px 13px; cursor: pointer; }
    button.active { background: #1769e0; border-color: #3b82f6; }
    #status { margin-left: auto; color: #aab7ca; font-size: 13px; }
    main { display: grid; place-items: center; min-height: calc(100vh - 68px); }
    img { display: block; max-width: 100%; max-height: calc(100vh - 72px);
          object-fit: contain; background: black; }
  </style>
</head>
<body>
  <header>
    <h1>RDK X5 · DECXIN Camera</h1>
    <button data-view="full" class="active">完整画面</button>
    <button data-view="left">左半画面</button>
    <button data-view="right">右半画面</button>
    <button id="fullscreen">全屏</button>
    <span id="status">连接中…</span>
  </header>
  <main><img id="camera" alt="Camera stream"></main>
  <script>
    const image = document.querySelector('#camera');
    const status = document.querySelector('#status');
    function setView(view) {
      document.querySelectorAll('[data-view]').forEach(
        b => b.classList.toggle('active', b.dataset.view === view));
      image.src = `/stream.mjpg?view=${view}&width=1280&fps=15&t=${Date.now()}`;
    }
    document.querySelectorAll('[data-view]').forEach(
      b => b.addEventListener('click', () => setView(b.dataset.view)));
    document.querySelector('#fullscreen').addEventListener('click', () => {
      if (image.requestFullscreen) image.requestFullscreen();
    });
    async function updateStatus() {
      try {
        const r = await fetch('/status', {cache: 'no-store'});
        const s = await r.json();
        status.textContent = s.opened
          ? `${s.width}×${s.height} · 源帧 ${s.frame_id}`
          : `相机未打开：${s.error || '未知错误'}`;
      } catch (_) { status.textContent = '状态连接失败'; }
    }
    setView('full');
    updateStatus();
    setInterval(updateStatus, 2000);
  </script>
</body>
</html>
"""


def device_index(device: str) -> int | str:
    if device.startswith("/dev/video") and device[10:].isdigit():
        return int(device[10:])
    if device.isdigit():
        return int(device)
    return device


class Camera:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.condition = threading.Condition()
        self.frame = None
        self.frame_id = 0
        self.opened = False
        self.error = ""
        self.source_width = 0
        self.source_height = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=3)

    def _capture_loop(self) -> None:
        cap = cv2.VideoCapture(device_index(self.args.device), cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.source_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.source_height)
        cap.set(cv2.CAP_PROP_FPS, self.args.source_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not cap.isOpened():
            self.error = f"无法打开 {self.args.device}，请确认没有被 ROS/OpenCV 占用"
            with self.condition:
                self.condition.notify_all()
            return

        self.opened = True
        failures = 0
        try:
            while not self.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    failures += 1
                    self.error = "读取图像失败"
                    if failures >= 30:
                        break
                    time.sleep(0.03)
                    continue
                failures = 0
                self.source_height, self.source_width = frame.shape[:2]
                with self.condition:
                    self.frame = frame
                    self.frame_id += 1
                    self.condition.notify_all()
        finally:
            self.opened = False
            cap.release()
            with self.condition:
                self.condition.notify_all()

    def wait_frame(self, last_id: int, timeout: float = 2.0):
        with self.condition:
            self.condition.wait_for(
                lambda: self.frame_id != last_id or not self.opened,
                timeout=timeout,
            )
            return self.frame_id, self.frame


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "RDKCamera/1.0"

    @property
    def camera(self) -> Camera:
        return self.server.camera  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args) -> None:
        print(f"[{self.address_string()}] {fmt % args}")

    def send_bytes(self, data: bytes, content_type: str, status=HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_bytes(INDEX_HTML.encode(), "text/html; charset=utf-8")
        elif parsed.path == "/status":
            payload = json.dumps(
                {
                    "opened": self.camera.opened,
                    "error": self.camera.error,
                    "width": self.camera.source_width,
                    "height": self.camera.source_height,
                    "frame_id": self.camera.frame_id,
                },
                ensure_ascii=False,
            ).encode()
            self.send_bytes(payload, "application/json; charset=utf-8")
        elif parsed.path == "/stream.mjpg":
            self.stream(parsed.query)
        else:
            self.send_error(HTTPStatus.NOT_FOUND)

    def stream(self, query: str) -> None:
        params = parse_qs(query)
        view = params.get("view", ["full"])[0]
        if view not in {"full", "left", "right"}:
            view = "full"
        width = max(320, min(1920, int(params.get("width", ["1280"])[0])))
        fps = max(1, min(30, int(params.get("fps", ["15"])[0])))
        quality = max(40, min(95, int(params.get("quality", ["80"])[0])))

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        last_id = -1
        next_send = 0.0
        try:
            while True:
                frame_id, frame = self.camera.wait_frame(last_id)
                if frame is None:
                    if not self.camera.opened:
                        break
                    continue
                last_id = frame_id
                now = time.monotonic()
                if now < next_send:
                    continue
                next_send = now + 1.0 / fps

                height, source_width = frame.shape[:2]
                if view == "left":
                    frame = frame[:, : source_width // 2]
                elif view == "right":
                    frame = frame[:, source_width // 2 :]

                if frame.shape[1] > width:
                    out_height = max(1, round(frame.shape[0] * width / frame.shape[1]))
                    frame = cv2.resize(frame, (width, out_height), interpolation=cv2.INTER_AREA)

                ok, encoded = cv2.imencode(
                    ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality]
                )
                if not ok:
                    continue
                data = encoded.tobytes()
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(f"Content-Length: {len(data)}\r\n\r\n".encode())
                self.wfile.write(data)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RDK X5 USB camera web viewer")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--source-width", type=int, default=4000)
    parser.add_argument("--source-height", type=int, default=1200)
    parser.add_argument("--source-fps", type=int, default=30)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8088)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    camera = Camera(args)
    camera.start()
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    server.camera = camera  # type: ignore[attr-defined]

    stopping = threading.Event()

    def stop_server(_signum=None, _frame=None) -> None:
        if not stopping.is_set():
            stopping.set()
            threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)
    print(f"Camera viewer: http://<RDK-IP>:{args.port}")
    try:
        server.serve_forever()
    finally:
        server.server_close()
        camera.stop()


if __name__ == "__main__":
    main()
