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
"""GJK and EPA collision detection functions."""

import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src.collision_core import Geom
from mujoco.mjx.third_party.mujoco_warp._src.types import mat43

FLOAT_MIN: float
FLOAT_MAX: float
MINVAL: float
MINVAL2: float
MAXVAL: float
MAXVAL2: float
MIN_DIST2: float
MIN_DIST3: float
MIN_DIST4: float
MIN_EPATOL: float
FACE_TOL: float
EDGE_TOL: float
INTERSECT_TOL: float

class GJKResult:
  separated: bool
  dist: float
  x1: wp.vec3
  x2: wp.vec3
  dim: int
  simplex: mat43
  simplex1: mat43
  simplex2: mat43
  simplex_index1: wp.vec4i
  simplex_index2: wp.vec4i
  index1: int
  index2: int

class Polytope:
  status: int
  vert: wp.array[wp.vec3]
  vert_index: wp.array[int]
  nvert: int
  center: wp.vec3
  face: wp.array[int]
  face_pr: wp.array[wp.vec3]
  face_norm2: wp.array[float]
  nface: int
  horizon: wp.array[int]
  nhorizon: int

class SupportPoint:
  point: wp.vec3
  cached_index: int
  vertex_index: int

@wp.func
def support(geom: Geom, geomtype: int, dir: wp.vec3) -> SupportPoint: ...
@wp.func
def gjk(
  tolerance: float,
  gjk_iterations: int,
  geom1: Geom,
  geom2: Geom,
  x1_0: wp.vec3,
  x2_0: wp.vec3,
  geomtype1: int,
  geomtype2: int,
  cutoff: float,
  is_discrete: bool,
) -> GJKResult: ...
@wp.func
def multicontact(
  polygon: wp.array[wp.vec3],
  clipped: wp.array[wp.vec3],
  plane_normal: wp.array[wp.vec3],
  plane_dist: wp.array[float],
  idx1: wp.array[int],
  idx2: wp.array[int],
  n1: wp.array[wp.vec3],
  n2: wp.array[wp.vec3],
  endvert: wp.array[wp.vec3],
  face1: wp.array[wp.vec3],
  face2: wp.array[wp.vec3],
  epa_vert: wp.array[wp.vec3],
  epa_vert_index: wp.array[int],
  epa_face: int,
  x1: wp.vec3,
  x2: wp.vec3,
  geom1: Geom,
  geom2: Geom,
  geomtype1: int,
  geomtype2: int,
) -> tuple[int, mat43, mat43, wp.vec4]: ...
@wp.func
def gjk_phase(
  tolerance: float,
  cutoff: float,
  gjk_iterations: int,
  geom1: Geom,
  geom2: Geom,
  geomtype1: int,
  geomtype2: int,
  x_1: wp.vec3,
  x_2: wp.vec3,
) -> tuple[bool, float, int, wp.vec3, wp.vec3, GJKResult, Geom, Geom]: ...
@wp.func
def epa_phase(
  tolerance: float,
  epa_iterations: int,
  result: GJKResult,
  geom1: Geom,
  geom2: Geom,
  geomtype1: int,
  geomtype2: int,
  vert: wp.array[wp.vec3],
  vert_index: wp.array[int],
  face: wp.array[int],
  face_pr: wp.array[wp.vec3],
  face_norm2: wp.array[float],
  horizon: wp.array[int],
  warn_overflow: bool,
  worldid: int,
  overflow_out: wp.array[int],
) -> tuple[float, int, wp.vec3, wp.vec3, int]: ...
@wp.func
def ccd(
  tolerance: float,
  cutoff: float,
  gjk_iterations: int,
  epa_iterations: int,
  geom1: Geom,
  geom2: Geom,
  geomtype1: int,
  geomtype2: int,
  x_1: wp.vec3,
  x_2: wp.vec3,
  vert: wp.array[wp.vec3],
  vert_index: wp.array[int],
  face: wp.array[int],
  face_pr: wp.array[wp.vec3],
  face_norm2: wp.array[float],
  horizon: wp.array[int],
  warn_overflow: bool,
  worldid: int,
  overflow_out: wp.array[int],
) -> tuple[float, int, wp.vec3, wp.vec3, int]: ...
