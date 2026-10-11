# Copyright 2026 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS.

from collections.abc import Sequence

import numpy as np
from numpy import typing as npt

from mujoco import MjData, MjModel, MjvCamera, MjvGeom, MjvOption, MjvPerturb

class Viewer:
  """Native Studio window and renderer. Create and release it on the same thread."""

  def __init__(
    self, title: str, width: int, height: int, graphics_mode_str: str, /
  ) -> None: ...
  def InitRenderer(self, model: MjModel, /) -> None: ...
  def NewFrame(self) -> bool: ...
  def Present(
    self,
    model: MjModel,
    data: MjData,
    perturb: MjvPerturb,
    camera: MjvCamera,
    vis_options: MjvOption,
    render_flags: Sequence[int] | npt.NDArray[np.uint8],
    extra_geoms: Sequence[MjvGeom] = (),
  ) -> None:
    """Render the model and present the frame with these render flags."""
  def UploadImage(
    self, tex_id: int, img: str | bytes, width: int, height: int, bpp: int, /
  ) -> int: ...
  def RenderToTexture(
    self,
    model: MjModel,
    data: MjData,
    cam: MjvCamera,
    width: int,
    height: int,
    tex_id: int,
    /,
  ) -> int:
    """Render RGB pixels to a texture; return the texture ID."""
  def GetDropFile(self) -> str: ...
  def GetImGuiContext(self) -> int: ...
  def GetImPlotContext(self) -> int: ...

def IsCrd() -> bool: ...
def IsCuda() -> bool: ...
