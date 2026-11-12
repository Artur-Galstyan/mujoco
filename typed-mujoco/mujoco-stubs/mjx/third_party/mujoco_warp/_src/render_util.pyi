# Copyright 2026 The Newton Developers
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
"""Rendering helpers: textures, output buffers and render context."""

from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import warp as wp

import mujoco
from mujoco.mjx.third_party.mujoco_warp._src.types import (
  RenderContext,
)

_FloatArray: TypeAlias = npt.NDArray[np.floating]
_IntArray: TypeAlias = npt.NDArray[np.integer]

def texture_pixels(
  mjm: mujoco.MjModel, tex_id: int
) -> npt.NDArray[np.uint8]: ...
def create_warp_texture(mjm: mujoco.MjModel, tex_id: int) -> wp.Texture2D: ...
@wp.func
def compute_ray(
  projection: int,
  fovy: float,
  sensorsize: wp.vec2,
  intrinsic: wp.vec4,
  img_w: int,
  img_h: int,
  px: int,
  py: int,
  znear: float,
) -> tuple[wp.vec3, wp.vec3]: ...
@wp.func
def pack_rgba_to_uint32(
  r: float, g: float, b: float, a: float
) -> wp.uint32: ...

unpack_rgb_kernel: wp.Kernel
extract_depth_kernel: wp.Kernel

def get_rgb(
  rc: RenderContext, camera_index: int, rgb_out: wp.array[wp.vec3]
) -> None: ...
def get_depth(
  rc: RenderContext,
  camera_index: int,
  depth_scale: float,
  depth_out: wp.array[float],
) -> None: ...
def get_segmentation(
  rc: RenderContext, camera_index: int, seg_out: wp.array[wp.vec2i]
) -> None: ...
def create_render_context(
  mjm: mujoco.MjModel,
  nworld: int = ...,
  cam_res: list[tuple[int, int]] | tuple[int, int] | None = ...,
  render_rgb: list[bool] | bool | None = ...,
  render_depth: list[bool] | bool | None = ...,
  render_seg: list[bool] | bool | None = ...,
  use_textures: bool = ...,
  use_fast_math: bool = ...,
  use_shadows: bool = ...,
  use_ambient_lighting: bool = ...,
  enabled_geom_groups: list[int] = ...,
  cam_active: list[bool] | list[str] | list[int] | None = ...,
  background_color: tuple[float, float, float, float] = ...,
  flex_render_smooth: bool = ...,
  use_precomputed_rays: bool = ...,
  render_skybox: bool = ...,
  enable_backface_culling: bool = ...,
  shadow_light_fraction: float = ...,
  samples_per_pixel: int = ...,
  enable_vertex_normals: bool = ...,
  enable_specular: bool = ...,
  enable_emission: bool = ...,
  enable_per_light_ambient: bool = ...,
  splat_position: _FloatArray | None = ...,
  splat_rotation: _FloatArray | None = ...,
  splat_scale: _FloatArray | None = ...,
  splat_rgba: _FloatArray | None = ...,
  splat_adr: _IntArray | None = ...,
  splat_group_id: _IntArray | None = ...,
  deterministic: bool = ...,
) -> RenderContext: ...
