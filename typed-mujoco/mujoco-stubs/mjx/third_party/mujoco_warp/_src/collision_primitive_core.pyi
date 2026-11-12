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
"""Core analytic collision functions for primitive geom pairs."""

from collections.abc import Sequence
from typing import overload

import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src.types import (
  mat53,
  mat113,
  vec5,
  vec11,
)

class vec8f:
  @overload
  def __init__(self) -> None: ...
  @overload
  def __init__(self, other: vec8f) -> None: ...
  @overload
  def __init__(self, *args: wp.float32 | float) -> None: ...
  @overload
  def __init__(self, args: Sequence[wp.float32]) -> None: ...

class mat23f:
  @overload
  def __init__(self) -> None: ...
  @overload
  def __init__(self, other: mat23f) -> None: ...
  @overload
  def __init__(self, *args: wp.float32 | float) -> None: ...
  @overload
  def __init__(self, *args: wp.vec3f) -> None: ...
  @overload
  def __init__(self, args: Sequence[wp.float32]) -> None: ...

class mat43f:
  @overload
  def __init__(self) -> None: ...
  @overload
  def __init__(self, other: mat43f) -> None: ...
  @overload
  def __init__(self, *args: wp.float32 | float) -> None: ...
  @overload
  def __init__(self, *args: wp.vec3f) -> None: ...
  @overload
  def __init__(self, args: Sequence[wp.float32]) -> None: ...

class mat83f:
  @overload
  def __init__(self) -> None: ...
  @overload
  def __init__(self, other: mat83f) -> None: ...
  @overload
  def __init__(self, *args: wp.float32 | float) -> None: ...
  @overload
  def __init__(self, *args: wp.vec3f) -> None: ...
  @overload
  def __init__(self, args: Sequence[wp.float32]) -> None: ...

@wp.func
def plane_sphere(
  plane_normal: wp.vec3,
  plane_pos: wp.vec3,
  sphere_pos: wp.vec3,
  sphere_radius: float,
) -> tuple[float, wp.vec3]: ...
@wp.func
def sphere_sphere(
  pos1: wp.vec3, radius1: float, pos2: wp.vec3, radius2: float
) -> tuple[float, wp.vec3, wp.vec3]: ...
@wp.func
def sphere_capsule(
  sphere_pos: wp.vec3,
  sphere_radius: float,
  capsule_pos: wp.vec3,
  capsule_axis: wp.vec3,
  capsule_radius: float,
  capsule_half_length: float,
) -> tuple[float, wp.vec3, wp.vec3]: ...
@wp.func
def capsule_capsule(
  cap1_pos: wp.vec3,
  cap1_axis: wp.vec3,
  cap1_radius: float,
  cap1_half_length: float,
  cap2_pos: wp.vec3,
  cap2_axis: wp.vec3,
  cap2_radius: float,
  cap2_half_length: float,
) -> tuple[wp.vec2, mat23f, mat23f]: ...
@wp.func
def plane_capsule(
  plane_normal: wp.vec3,
  plane_pos: wp.vec3,
  capsule_pos: wp.vec3,
  capsule_axis: wp.vec3,
  capsule_radius: float,
  capsule_half_length: float,
) -> tuple[wp.vec2, mat23f, wp.mat33]: ...
@wp.func
def plane_ellipsoid(
  plane_normal: wp.vec3,
  plane_pos: wp.vec3,
  ellipsoid_pos: wp.vec3,
  ellipsoid_rot: wp.mat33,
  ellipsoid_size: wp.vec3,
) -> tuple[float, wp.vec3, wp.vec3]: ...
@wp.func
def plane_box(
  plane_normal: wp.vec3,
  plane_pos: wp.vec3,
  box_pos: wp.vec3,
  box_rot: wp.mat33,
  box_size: wp.vec3,
) -> tuple[vec8f, mat83f, wp.vec3]: ...
@wp.func
def sphere_cylinder(
  sphere_pos: wp.vec3,
  sphere_radius: float,
  cylinder_pos: wp.vec3,
  cylinder_axis: wp.vec3,
  cylinder_radius: float,
  cylinder_half_height: float,
) -> tuple[float, wp.vec3, wp.vec3]: ...
@wp.func
def plane_cylinder(
  plane_normal: wp.vec3,
  plane_pos: wp.vec3,
  cylinder_center: wp.vec3,
  cylinder_axis: wp.vec3,
  cylinder_radius: float,
  cylinder_half_height: float,
) -> tuple[wp.vec4, mat43f, wp.vec3]: ...
@wp.func
def box_box(
  box1_pos: wp.vec3,
  box1_rot: wp.mat33,
  box1_size: wp.vec3,
  box2_pos: wp.vec3,
  box2_rot: wp.mat33,
  box2_size: wp.vec3,
  margin: float,
) -> tuple[int, vec8f, mat83f, wp.vec3]: ...
@wp.func
def sphere_box(
  sphere_pos: wp.vec3,
  sphere_radius: float,
  box_pos: wp.vec3,
  box_rot: wp.mat33,
  box_size: wp.vec3,
) -> tuple[float, wp.vec3, wp.vec3]: ...
@wp.func
def capsule_box(
  capsule_pos: wp.vec3,
  capsule_axis: wp.vec3,
  capsule_radius: float,
  capsule_half_length: float,
  box_pos: wp.vec3,
  box_rot: wp.mat33,
  box_size: wp.vec3,
) -> tuple[wp.vec2, mat23f, mat23f]: ...
@wp.func
def sphere_triangle(
  sphere_pos: wp.vec3,
  sphere_radius: float,
  t1: wp.vec3,
  t2: wp.vec3,
  t3: wp.vec3,
  tri_radius: float,
  margin: float,
) -> tuple[float, wp.vec3, wp.vec3]: ...
@wp.func
def box_triangle(
  box_pos: wp.vec3,
  box_rot: wp.mat33,
  box_size: wp.vec3,
  t1: wp.vec3,
  t2: wp.vec3,
  t3: wp.vec3,
  tri_radius: float,
  margin: float,
) -> tuple[vec11, mat113, mat113]: ...
@wp.func
def tri_weights(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.vec3: ...
@wp.func
def capsule_triangle(
  capsule_pos: wp.vec3,
  capsule_axis: wp.vec3,
  capsule_radius: float,
  capsule_half_length: float,
  t1: wp.vec3,
  t2: wp.vec3,
  t3: wp.vec3,
  tri_radius: float,
  margin: float,
) -> tuple[vec5, mat53, mat53]: ...
@wp.func
def cylinder_triangle(
  cylinder_pos: wp.vec3,
  cylinder_axis: wp.vec3,
  cylinder_radius: float,
  cylinder_half_height: float,
  t1: wp.vec3,
  t2: wp.vec3,
  t3: wp.vec3,
  tri_radius: float,
) -> tuple[wp.vec2, mat23f, mat23f]: ...
@wp.func
def capsule_tetrahedron(
  pa: wp.vec3,
  pb: wp.vec3,
  radius: float,
  v0: wp.vec3,
  v1: wp.vec3,
  v2: wp.vec3,
  v3: wp.vec3,
  tet_radius: float,
) -> tuple[bool, float, wp.vec3, wp.vec3]: ...
