# Copyright 2026 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS.
"""Simulation-agnostic native viewer for MuJoCo models."""

from typing import Any

import mujoco
from mujoco.studio import endpoints, ux, viewer_protocol

class NativeViewer(viewer_protocol.Viewer):
  """Simulation-agnostic native viewer for MuJoCo models."""

  def __init__(
    self,
    config: viewer_protocol.ViewerConfig,
    endpoint: endpoints.ViewerEndpoint,
    *,
    model: mujoco.MjModel | None = None,
    plugins: list[Any] | None = None,
    camera: mujoco.MjvCamera | None = None,
    vis_options: mujoco.MjvOption | None = None,
    perturb: mujoco.MjvPerturb | None = None,
    render_flags: ux.RenderFlags | None = None,
    extra_geoms: list[mujoco.MjvGeom] | None = None,
  ) -> None: ...
  def prepare_next_frame(self) -> bool:
    """Advance to the next frame; return False when the window is closed."""
  def sync(self) -> None:
    """Render the scene and present it to the window."""
  def close(self) -> None:
    """Close the viewer and destroy the renderer."""
  def get_drop_file(self) -> str:
    """Return the path of the file dropped into the window, or ''."""
  def upload_image(
    self, tex_id: int, img: str | bytes, width: int, height: int, bpp: int
  ) -> int:
    """Upload an image to the backend for GUI rendering."""
