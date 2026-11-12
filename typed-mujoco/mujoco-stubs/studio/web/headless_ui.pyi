# Copyright 2026 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS.

class HeadlessUi:
  """Stream ImGui frames to a browser connected over NetImgui."""

  def __init__(self, title: str, assets_dir: str = '') -> None: ...
  def new_frame(self) -> bool:
    """Start a frame; return False if no browser controls the UI."""
  def end_frame(self) -> None:
    """End the frame and queue it for the browser."""
  def start_session(self, handshake: bytes) -> bytes:
    """Start streaming to a new browser; return the reply to send back."""
  def end_session(self) -> None:
    """Stop streaming; new_frame returns False until the next session."""
  def push_input(self, message: bytes) -> bool:
    """Queue an input message from the browser for the next frame."""
  def take_messages(self) -> list[bytes]:
    """Get the messages to send to the browser now."""
  def last_frame_index(self) -> int:
    """Get the index of the last frame that take_messages returned."""
  def get_context(self) -> int: ...
  def get_implot_context(self) -> int: ...
  def upload_image(
    self, tex_id: int, pixels: bytes, width: int, height: int, bpp: int, /
  ) -> int: ...
