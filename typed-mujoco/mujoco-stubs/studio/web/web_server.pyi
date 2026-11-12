# Copyright 2026 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS.
"""Single-port HTTP and WebSocket server of the Studio web viewer."""

import logging
import queue
import socket
from typing import Protocol

logger: logging.Logger

class UiSession(Protocol):
  """The headless UI as the /ui stream sees it."""

  def start_session(self, handshake: bytes) -> bytes:
    """Start a session and return the reply to the browser's handshake."""
  def end_session(self) -> None:
    """End the session."""
  def push_input(self, message: bytes) -> bool:
    """Queue browser input for the next frame; False if it is not input."""
  def take_messages(self) -> list[bytes]:
    """Return the messages to send to the browser now, possibly none."""
  def last_frame_index(self) -> int:
    """Return the index of the last draw frame take_messages returned."""

def bind_public_socket(host: str, port: int = 0) -> socket.socket:
  """Bind (but do not listen on) the public HTTP listening socket."""

class WebServer:
  """The Web Viewer's single-port server."""

  http_sock: socket.socket
  ui: UiSession | None
  static_files_dir: str | None
  mjb_data: bytes | None
  drop_queue: queue.Queue[dict[str, bytes]] | None
  model_cache_bytes: int
  stop_reason: str
  def __init__(
    self,
    http_sock: socket.socket,
    ui: UiSession | None = None,
    static_files_dir: str | None = None,
    mjb_data: bytes | None = None,
    drop_queue: queue.Queue[dict[str, bytes]] | None = None,
  ) -> None: ...
  def update_state(self, payload: bytes) -> None:
    """Publish the latest state payload."""
  def notify_ui_frame(self) -> None:
    """Tell the /ui stream that the UI has a new frame."""
  def update_model(self, mjb_data: bytes, crc32: int | None = None) -> None:
    """Make ``mjb_data`` the current model, served at /model."""
  def add_model(self, mjb_data: bytes, crc32: int | None = None) -> int:
    """Cache a bundle and tell browsers to prefetch it; return its crc32."""
  def cached_model(self, crc32: int) -> bytes | None:
    """Return the cached bundle with this crc32, or None."""
  @property
  def viewer_count(self) -> int:
    """Number of browsers currently connected to the /state stream."""
  @property
  def models_generation(self) -> int:
    """Count of changes to the prefetch line."""
  def prefetch_line(self) -> str:
    """Return the prefetch line for browsers."""
  def set_theme(self, theme: str) -> None:
    """Record a UI theme picked by the controller browser."""
  def pop_theme(self) -> str | None:
    """Return and clear any UI theme picked by the controller browser."""
  def set_client_camera(self, camera: bytes) -> None:
    """Record the latest camera sent by the controlling browser."""
  def take_client_camera(self) -> bytes | None:
    """Return the camera the controller last sent, once; else None."""
  def start(self) -> None:
    """Start the server on a background daemon thread."""
  def stop(self, reason: str = '') -> None:
    """Stop the server thread."""
