# Copyright 2025 The Physics-Next Project Developers
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
"""Signed distance field collision functions and narrowphase kernel."""

import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src.collision_core import (
  CollisionContext,
)
from mujoco.mjx.third_party.mujoco_warp._src.types import (
  Data,
  Model,
  vec8,
  vec8i,
  vec_pluginattr,
)

class OptimizationParams:
  rel_mat: wp.mat33
  rel_pos: wp.vec3
  attr1: vec_pluginattr
  attr2: vec_pluginattr

class AABB:
  min: wp.vec3
  max: wp.vec3

class VolumeData:
  center: wp.vec3
  half_size: wp.vec3
  oct_aabb: wp.array[wp.vec3]
  oct_child: wp.array[vec8i]
  oct_coeff: wp.array[vec8]
  root: int
  valid: bool

class MeshData:
  nmeshface: int
  mesh_vertadr: wp.array[int]
  mesh_vert: wp.array[wp.vec3]
  mesh_faceadr: wp.array[int]
  mesh_face: wp.array[wp.vec3i]
  data_id: int
  pos: wp.vec3
  mat: wp.mat33
  size: wp.vec3
  pnt: wp.vec3
  vec: wp.vec3
  valid: bool

@wp.func
def get_sdf_params(
  oct_child: wp.array[vec8i],
  oct_aabb: wp.array[wp.vec3],
  oct_coeff: wp.array[vec8],
  mesh_octadr: wp.array[int],
  plugin: wp.array[int],
  plugin_attr: wp.array[vec_pluginattr],
  g_type: int,
  g_size: wp.vec3,
  plugin_id: int,
  mesh_id: int,
) -> tuple[vec_pluginattr, int, VolumeData, MeshData]: ...
@wp.func
def transform_aabb(
  aabb_pos: wp.vec3, aabb_size: wp.vec3, pos: wp.vec3, ori: wp.mat33
) -> AABB: ...
@wp.func
def radial_field(a: wp.vec3, x: wp.vec3, size: wp.vec3) -> wp.vec3: ...
@wp.func
def sphere(p: wp.vec3, size: wp.vec3) -> float: ...
@wp.func
def box(p: wp.vec3, size: wp.vec3) -> float: ...
@wp.func
def ellipsoid(p: wp.vec3, size: wp.vec3) -> float: ...
@wp.func
def capsule(p: wp.vec3, size: wp.vec3) -> float: ...
@wp.func
def cylinder(p: wp.vec3, size: wp.vec3) -> float: ...
@wp.func
def grad_sphere(p: wp.vec3) -> wp.vec3: ...
@wp.func
def grad_box(p: wp.vec3, size: wp.vec3) -> wp.vec3: ...
@wp.func
def grad_ellipsoid(p: wp.vec3, size: wp.vec3) -> wp.vec3: ...
@wp.func
def grad_capsule(p: wp.vec3, size: wp.vec3) -> wp.vec3: ...
@wp.func
def grad_cylinder(p: wp.vec3, size: wp.vec3) -> wp.vec3: ...
@wp.func
def user_sdf(p: wp.vec3, attr: vec_pluginattr, sdf_type: int) -> float: ...
@wp.func
def user_sdf_grad(
  p: wp.vec3, attr: vec_pluginattr, sdf_type: int
) -> wp.vec3: ...
@wp.func
def find_oct(
  oct_child: wp.array[vec8i],
  oct_aabb: wp.array[wp.vec3],
  p: wp.vec3,
  grad: bool,
  root: int,
) -> tuple[int, tuple[vec8, vec8, vec8]]: ...
@wp.func
def box_project(
  center: wp.vec3, half_size: wp.vec3, xyz: wp.vec3
) -> tuple[float, wp.vec3]: ...
@wp.func
def sample_volume_sdf(xyz: wp.vec3, volume_data: VolumeData) -> float: ...
@wp.func
def sample_volume_grad(xyz: wp.vec3, volume_data: VolumeData) -> wp.vec3: ...
@wp.func
def sdf(
  type: int,
  p: wp.vec3,
  attr: vec_pluginattr,
  sdf_type: int,
  volume_data: VolumeData,
  mesh_data: MeshData,
) -> float: ...
@wp.func
def sdf_grad(
  type: int,
  p: wp.vec3,
  attr: vec_pluginattr,
  sdf_type: int,
  volume_data: VolumeData,
  mesh_data: MeshData,
) -> wp.vec3: ...
@wp.func
def clearance(
  type1: int,
  p1: wp.vec3,
  p2: wp.vec3,
  s1: vec_pluginattr,
  s2: vec_pluginattr,
  sdf_type1: int,
  sdf_type2: int,
  sfd_intersection: bool,
  volume_data1: VolumeData,
  volume_data2: VolumeData,
  mesh_data1: MeshData,
  mesh_data2: MeshData,
) -> float: ...
@wp.func
def compute_grad(
  type1: int,
  p1: wp.vec3,
  p2: wp.vec3,
  params: OptimizationParams,
  sdf_type1: int,
  sdf_type2: int,
  sfd_intersection: bool,
  volume_data1: VolumeData,
  volume_data2: VolumeData,
  mesh_data1: MeshData,
  mesh_data2: MeshData,
) -> tuple[wp.vec3, float]: ...
@wp.func
def gradient_step(
  type1: int,
  x: wp.vec3,
  params: OptimizationParams,
  sdf_type1: int,
  sdf_type2: int,
  niter: int,
  sfd_intersection: bool,
  alpha0: float,
  volume_data1: VolumeData,
  volume_data2: VolumeData,
  mesh_data1: MeshData,
  mesh_data2: MeshData,
) -> tuple[float, wp.vec3]: ...
@wp.func
def gradient_descent(
  type1: int,
  x0_initial: wp.vec3,
  attr1: vec_pluginattr,
  attr2: vec_pluginattr,
  pos1: wp.vec3,
  rot1: wp.mat33,
  pos2: wp.vec3,
  rot2: wp.mat33,
  sdf_type1: int,
  sdf_type2: int,
  sdf_iterations: int,
  linesearch_alpha0: float,
  volume_data1: VolumeData,
  volume_data2: VolumeData,
  mesh_data1: MeshData,
  mesh_data2: MeshData,
) -> tuple[float, wp.vec3, wp.vec3]: ...
def sdf_narrowphase(m: Model, d: Data, ctx: CollisionContext) -> None: ...
