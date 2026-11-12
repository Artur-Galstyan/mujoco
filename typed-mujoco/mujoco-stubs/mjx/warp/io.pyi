# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""I/O functions for MJX Warp."""

from typing import TypedDict

import numpy as np
from numpy import typing as npt
from typing_extensions import Unpack

import mujoco
from mujoco.mjx.warp.render_context import RenderContext

class _RenderContextKwargs(TypedDict, total=False):
  cam_res: list[tuple[int, int]] | tuple[int, int] | None
  render_rgb: list[bool] | bool | None
  render_depth: list[bool] | bool | None
  render_seg: list[bool] | bool | None
  use_textures: bool
  use_fast_math: bool
  use_shadows: bool
  use_ambient_lighting: bool
  enabled_geom_groups: list[int]
  cam_active: list[bool] | list[str] | list[int] | None
  background_color: tuple[float, float, float, float]
  flex_render_smooth: bool
  use_precomputed_rays: bool
  render_skybox: bool
  enable_backface_culling: bool
  shadow_light_fraction: float
  samples_per_pixel: int
  enable_vertex_normals: bool
  enable_specular: bool
  enable_emission: bool
  enable_per_light_ambient: bool
  splat_position: npt.NDArray[np.float32 | np.float64] | None
  splat_rotation: npt.NDArray[np.float32 | np.float64] | None
  splat_scale: npt.NDArray[np.float32 | np.float64] | None
  splat_rgba: npt.NDArray[np.float32 | np.float64] | None
  splat_adr: npt.NDArray[np.int32 | np.int64] | None
  splat_group_id: npt.NDArray[np.int32 | np.int64] | None
  deterministic: bool

def create_render_context(
  mjm: mujoco.MjModel,
  nworld: int,
  devices: list[str | None] | None = ...,
  **kwargs: Unpack[_RenderContextKwargs],
) -> RenderContext:
  """Creates a render context using mujoco_warp.create_render_context."""
