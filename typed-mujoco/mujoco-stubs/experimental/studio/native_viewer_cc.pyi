# Copyright 2026 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

from collections.abc import Sequence

from mujoco import (
  MjData,
  MjModel,
  MjvCamera,
  MjvGeom,
  MjvOption,
  MjvPerturb,
  mjrf,
)

class ImguiBridge:
  def __init__(self, ctx: mjrf.Context, scene: mjrf.Scene) -> None: ...
  def update(self) -> None: ...
  def upload_image(
    self, tex_id: int, img: str | bytes, width: int, height: int, bpp: int
  ) -> int: ...

class ModelScene:
  def __init__(
    self, ctx: mjrf.Context, scene: mjrf.Scene, model: MjModel
  ) -> None: ...
  def sync(
    self,
    width: int,
    height: int,
    model: MjModel,
    data: MjData,
    perturb: MjvPerturb,
    camera: MjvCamera,
    vis_options: MjvOption,
    extra_geoms: Sequence[MjvGeom],
  ) -> None: ...
  def get_headlight_intensity(self) -> float: ...

def render_to_pixels(graphics_mode: int) -> bool: ...
