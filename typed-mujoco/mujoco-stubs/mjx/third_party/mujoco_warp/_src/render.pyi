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
"""Batched ray tracing renderer."""

import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src.types import (
  Data,
  Model,
  RenderContext,
)

MAX_SHININESS: float
DEFAULT_MAT_SHININESS_EXPONENT: float
DEFAULT_MAT_SPECULAR: float
DEFAULT_MAT_EMISSION: float
NO_LIGHT_AMBIENT_FALLBACK: float

@wp.func
def ray_splat(
  position: wp.vec3,
  rotation: wp.quat,
  scale: wp.vec3,
  opacity: float,
  pnt: wp.vec3,
  vec: wp.vec3,
  min_distance: float,
  max_distance: float,
) -> tuple[float, float]: ...
@wp.func
def shade_splats(
  splat_position: wp.array[wp.vec3],
  splat_rotation: wp.array[wp.quat],
  splat_scale: wp.array[wp.vec3],
  splat_rgba: wp.array[wp.vec4],
  bvh_id: wp.uint64,
  group_root: int,
  ray_origin: wp.vec3,
  ray_direction: wp.vec3,
  max_distance: float,
) -> tuple[wp.vec3, float, float]: ...
@wp.func
def sample_skybox(
  skybox_tex: wp.Texture2D, face_width_inv: float, ray_dir_world: wp.vec3
) -> wp.vec3: ...
@wp.func
def vertex_normal(n: wp.vec3, face: wp.vec3) -> wp.vec3: ...
def render(m: Model, d: Data, rc: RenderContext) -> None: ...
