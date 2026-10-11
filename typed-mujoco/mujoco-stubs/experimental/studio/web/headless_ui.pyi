# Copyright 2026 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS.

class HeadlessUi:
  """Stream ImGui frames to a browser connected over NetImgui."""

  def __init__(
    self, title: str, port: int = 8888, assets_dir: str = ''
  ) -> None: ...
  def new_frame(self) -> bool:
    """Start a frame; return False if the browser is not connected."""
  def end_frame(self) -> None: ...
  def get_context(self) -> int: ...
  def get_implot_context(self) -> int: ...
  def upload_image(
    self, tex_id: int, pixels: bytes, width: int, height: int, bpp: int, /
  ) -> int: ...
