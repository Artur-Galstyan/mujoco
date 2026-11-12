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
"""Passive forces: springs, dampers, fluid forces and flex elasticity."""

from typing import Literal, TypeAlias

import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src.types import (
  Data,
  Model,
  mat43,
  mat63,
  mat66,
  vec6,
)

_EfmContact: TypeAlias = tuple[
  wp.array[int],
  wp.array[float],
  wp.array[float],
  wp.array[float],
  wp.array[int],
]
_FlexScratch: TypeAlias = tuple[wp.array[wp.vec3], wp.array[wp.vec3]]

@wp.func
def geom_semiaxes(size: wp.vec3, geom_type: int) -> wp.vec3: ...
@wp.func
def ellipsoid_max_moment(size: wp.vec3, dir: int) -> float: ...
@wp.func
def stretch_edge_endpoints(
  dim: int,
) -> wp.Matrix[int, Literal[6], Literal[2]]: ...
@wp.func
def stretch_edge_vectors(vert_xpos: mat43, dim: int) -> mat63: ...
@wp.func
def stretch_elongation(
  flex_elemedge: wp.array[int],
  flexedge_length0: wp.array[float],
  plastic_in: wp.array[float],
  flexedge_length_in: wp.array[float],
  worldid: int,
  elemedge_adr: int,
  edge_adr: int,
  padr: int,
  nedge: int,
) -> vec6: ...
@wp.func
def stretch_metric(
  flex_stiffness: wp.array[float], stiffness_adr: int, nedge: int
) -> mat66: ...
@wp.func
def stretch_tension(metric: mat66, elongation: vec6, nedge: int) -> vec6: ...
@wp.func
def stretch_elasticity(
  flex_stiffness: wp.array[float],
  stiffness_adr: int,
  elongation: vec6,
  nedge: int,
) -> tuple[mat66, vec6]: ...
@wp.func
def stretch_stiffness(
  flex_elemedge: wp.array[int],
  flexedge_length0: wp.array[float],
  flex_stiffness: wp.array[float],
  plastic_in: wp.array[float],
  flexedge_length_in: wp.array[float],
  worldid: int,
  stiffness_adr: int,
  elemedge_adr: int,
  edge_adr: int,
  padr: int,
  nedge: int,
) -> tuple[mat66, vec6]: ...
@wp.func
def stretch_stiffness_block(
  metric: mat66,
  tension: vec6,
  edgevec: mat63,
  dim: int,
  i: int,
  j: int,
  scale: float,
) -> wp.mat33: ...
@wp.func
def snh_project(
  flex_vert0: wp.array[wp.vec3],
  flex_stiffness: wp.array[float],
  stiffness_adr: int,
  vbase: int,
  vert: wp.vec4i,
  size: wp.vec3,
  edgevec: mat63,
  elongation: vec6,
) -> tuple[bool, wp.mat33, mat43, wp.mat33, wp.vec3, wp.vec3]: ...
@wp.func
def snh_projected_block(
  rotation: wp.mat33,
  gradient: mat43,
  stretch: wp.mat33,
  symmetric: wp.vec3,
  skew: wp.vec3,
  i: int,
  j: int,
) -> wp.mat33: ...
@wp.func
def flex_gather_vert(
  body_rootid: wp.array[int],
  body_weldid: wp.array[int],
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  body_simple: wp.array[int],
  dof_parentid: wp.array[int],
  flex_vertbodyid: wp.array[int],
  subtree_com_in: wp.array[wp.vec3],
  cdof_in: wp.array[wp.spatial_vectorf],
  flexvert_xpos_in: wp.array[wp.vec3],
  vec_in: wp.array[float],
  worldid: int,
  gvert: int,
) -> wp.vec3: ...
@wp.func
def flex_scatter_vert(
  body_rootid: wp.array[int],
  body_weldid: wp.array[int],
  body_dofnum: wp.array[int],
  body_dofadr: wp.array[int],
  body_simple: wp.array[int],
  dof_parentid: wp.array[int],
  flex_vertbodyid: wp.array[int],
  subtree_com_in: wp.array[wp.vec3],
  cdof_in: wp.array[wp.spatial_vectorf],
  flexvert_xpos_in: wp.array[wp.vec3],
  worldid: int,
  gvert: int,
  f_world: wp.vec3,
  scale: float,
  res_out: wp.array[float],
) -> None: ...
def eval_rod_chains(m: Model, d: Data) -> None: ...
@wp.func
def rod_stencil(
  flex_bending: wp.array[float],
  flexvert_xpos_in: wp.array[wp.vec3],
  worldid: int,
  vbase: int,
  adr: int,
) -> tuple[
  bool, wp.vec4i, mat43, mat43, mat43, mat43, wp.vec4, wp.mat44, wp.mat44
]: ...
@wp.func
def rod_stencil_jacobian(
  J0: mat43, J1: mat43, J2: mat43, J3: mat43, i: int
) -> mat43: ...
def flex_gather(
  m: Model,
  d: Data,
  res: wp.array[wp.vec3],
  vec: wp.array[float],
  skip: wp.array[bool] | None = ...,
) -> None:
  """Gathers generalized vector into world-space flex vertex coordinates."""

def flex_scatter(
  m: Model,
  d: Data,
  res: wp.array[float],
  vec: wp.array[wp.vec3],
  scale: float = ...,
  skip: wp.array[bool] | None = ...,
) -> None:
  """Scatters world-space flex vertex forces into generalized coordinates."""

def flex_hessian(m: Model, d: Data) -> None:
  """Caches the unscaled Cartesian stretch Hessian for all flexes."""

def flex_hessian_mul(
  m: Model,
  d: Data,
  res: wp.array[wp.vec3],
  vec: wp.array[wp.vec3],
  s1_in: float = ...,
  s2_in: float = ...,
  use_timestep: bool = ...,
  non_simple_only: bool = ...,
  snh_only: bool = ...,
  skip: wp.array[bool] | None = ...,
) -> None: ...
def flex_stretch_scratch(m: Model, d: Data) -> _FlexScratch:
  """Allocates the per-vertex scratch arrays used by flex_stretch_mul."""

def flex_stretch_mul(
  m: Model,
  d: Data,
  res: wp.array[float],
  vec: wp.array[float],
  s1_in: float = ...,
  s2_in: float = ...,
  use_timestep: bool = ...,
  non_simple_only: bool = ...,
  snh_only: bool = ...,
  skip: wp.array[bool] | None = ...,
  scratch: tuple[wp.array[wp.vec3], wp.array[wp.vec3]] | None = ...,
) -> None:
  """Adds (s1 + s2*flex_damping) * J^T * K_stretch * J * vec to res."""

def flex_bend_mul(
  m: Model,
  d: Data,
  res: wp.array[float],
  vec: wp.array[float],
  s1_in: float = ...,
  s2_in: float = ...,
  use_timestep: bool = ...,
  non_simple_only: bool = ...,
  skip: wp.array[bool] | None = ...,
) -> None:
  """Adds (s1 + s2*flex_damping) * J^T * K_bend * J * vec to res."""

def flex_rod_mul(
  m: Model,
  d: Data,
  res: wp.array[float],
  vec: wp.array[float],
  s1_in: float = ...,
  s2_in: float = ...,
  use_timestep: bool = ...,
  non_simple_only: bool = ...,
  skip: wp.array[bool] | None = ...,
) -> None: ...
def build_efm_contact(m: Model, d: Data, rebuild: bool = ...) -> _EfmContact:
  """Allocates and builds rank-1 passive flex contact metric terms."""

def passive(m: Model, d: Data) -> None:
  """Adds all passive forces."""
