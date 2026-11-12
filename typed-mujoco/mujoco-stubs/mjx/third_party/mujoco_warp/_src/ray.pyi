# Copyright 2025 The Newton Developers
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
"""Ray casting against geoms, meshes, flexes and the scene."""

import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src.types import (
  Data,
  Model,
  RenderContext,
  vec6,
)

RAY_TOL_REL: float
RAY_TOL_ABS: float

@wp.func
def ray_plane(
  pos: wp.vec3, mat: wp.mat33, size: wp.vec3, pnt: wp.vec3, vec: wp.vec3
) -> tuple[float, wp.vec3]: ...
@wp.func
def ray_sphere(
  pos: wp.vec3, dist_sqr: float, pnt: wp.vec3, vec: wp.vec3
) -> tuple[float, wp.vec3]: ...
@wp.func
def ray_capsule(
  pos: wp.vec3, mat: wp.mat33, size: wp.vec3, pnt: wp.vec3, vec: wp.vec3
) -> tuple[float, wp.vec3]: ...
@wp.func
def ray_ellipsoid(
  pos: wp.vec3, mat: wp.mat33, size: wp.vec3, pnt: wp.vec3, vec: wp.vec3
) -> tuple[float, wp.vec3]: ...
@wp.func
def ray_cylinder(
  pos: wp.vec3, mat: wp.mat33, size: wp.vec3, pnt: wp.vec3, vec: wp.vec3
) -> tuple[float, wp.vec3]: ...
@wp.func
def ray_box(
  pos: wp.vec3, mat: wp.mat33, size: wp.vec3, pnt: wp.vec3, vec: wp.vec3
) -> tuple[float, vec6, wp.vec3]: ...
@wp.func
def ray_hfield(
  geom_dataid: wp.array[int],
  hfield_size: wp.array[wp.vec4],
  hfield_nrow: wp.array[int],
  hfield_ncol: wp.array[int],
  hfield_adr: wp.array[int],
  hfield_data: wp.array[float],
  pos: wp.vec3,
  mat: wp.mat33,
  pnt: wp.vec3,
  vec: wp.vec3,
  id: int,
  worldid: int,
) -> tuple[float, wp.vec3]: ...
@wp.func
def ray_mesh(
  nmeshface: int,
  mesh_vertadr: wp.array[int],
  mesh_faceadr: wp.array[int],
  mesh_vert: wp.array[wp.vec3],
  mesh_face: wp.array[wp.vec3i],
  data_id: int,
  pos: wp.vec3,
  mat: wp.mat33,
  size: wp.vec3,
  pnt: wp.vec3,
  vec: wp.vec3,
) -> tuple[float, wp.vec3]: ...
@wp.func
def ray_mesh_with_bvh(
  mesh_bvh_id: wp.array[wp.uint64],
  mesh_geom_id: int,
  pos: wp.vec3,
  mat: wp.mat33,
  pnt: wp.vec3,
  vec: wp.vec3,
  max_t: float,
  cull_backfaces: bool,
) -> tuple[float, wp.vec3, float, float, int, int]: ...
@wp.func
def ray_mesh_with_bvh_anyhit(
  mesh_bvh_id: wp.array[wp.uint64],
  mesh_geom_id: int,
  pos: wp.vec3,
  mat: wp.mat33,
  pnt: wp.vec3,
  vec: wp.vec3,
  max_t: float,
) -> bool: ...
@wp.func
def ray_flex_with_bvh(
  flex_bvh_id: wp.array[wp.uint64],
  flexid: int,
  group_root: int,
  pnt: wp.vec3,
  vec: wp.vec3,
  max_t: float,
) -> tuple[float, wp.vec3, float, float, int]: ...
@wp.func
def ray_flex_with_bvh_anyhit(
  flex_bvh_id: wp.array[wp.uint64],
  flexid: int,
  group_root: int,
  pnt: wp.vec3,
  vec: wp.vec3,
  max_t: float,
) -> bool: ...
@wp.func
def ray_geom(
  pos: wp.vec3,
  mat: wp.mat33,
  size: wp.vec3,
  pnt: wp.vec3,
  vec: wp.vec3,
  geomtype: int,
) -> tuple[float, wp.vec3]: ...
def ray(
  m: Model,
  d: Data,
  pnt: wp.array[wp.vec3],
  vec: wp.array[wp.vec3],
  geomgroup: vec6 | None = ...,
  flg_static: bool = ...,
  bodyexclude: int = ...,
  rc: RenderContext | None = ...,
) -> tuple[wp.array[float], wp.array[int], wp.array[wp.vec3]]: ...
def rays(
  m: Model,
  d: Data,
  pnt: wp.array[wp.vec3],
  vec: wp.array[wp.vec3],
  geomgroup: vec6,
  flg_static: bool,
  bodyexclude: wp.array[int],
  dist: wp.array[float],
  geomid: wp.array[int],
  normal: wp.array[wp.vec3],
  rc: RenderContext | None = ...,
) -> None: ...
