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
"""Core collision types and utilities shared across collision modules."""

import dataclasses
from typing import TypeVar

import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src.types import (
  mat63,
  vec5,
)

_T = TypeVar('_T')

class Geom:
  """Geom properties for pairwise collision detection."""

  pos: wp.vec3
  rot: wp.mat33
  normal: wp.vec3
  size: wp.vec3
  margin: float
  polyvert: mat63
  vertadr: int
  vertnum: int
  vert: wp.array[wp.vec3]
  graphadr: int
  graph: wp.array[int]
  mesh_polynum: int
  mesh_polyadr: int
  mesh_polynormal: wp.array[wp.vec3]
  mesh_polyvertadr: wp.array[int]
  mesh_polyvertnum: wp.array[int]
  mesh_polyvert: wp.array[int]
  mesh_polymapadr: wp.array[int]
  mesh_polymapnum: wp.array[int]
  mesh_polymap: wp.array[int]
  index: int

@wp.func
def geom_collision_pair_from_types(
  geom_dataid: wp.array[int],
  geom_size: wp.array[wp.vec3],
  mesh_vertadr: wp.array[int],
  mesh_vertnum: wp.array[int],
  mesh_graphadr: wp.array[int],
  mesh_vert: wp.array[wp.vec3],
  mesh_graph: wp.array[int],
  mesh_polynum: wp.array[int],
  mesh_polyadr: wp.array[int],
  mesh_polynormal: wp.array[wp.vec3],
  mesh_polyvertadr: wp.array[int],
  mesh_polyvertnum: wp.array[int],
  mesh_polyvert: wp.array[int],
  mesh_polymapadr: wp.array[int],
  mesh_polymapnum: wp.array[int],
  mesh_polymap: wp.array[int],
  geom_xpos_in: wp.array[wp.vec3],
  geom_xmat_in: wp.array[wp.mat33],
  geom_type1: int,
  geom_type2: int,
  geoms: wp.vec2i,
  worldid: int,
) -> tuple[Geom, Geom]: ...
@wp.func
def geom_collision_pair(
  geom_type: wp.array[int],
  geom_dataid: wp.array[int],
  geom_size: wp.array[wp.vec3],
  mesh_vertadr: wp.array[int],
  mesh_vertnum: wp.array[int],
  mesh_graphadr: wp.array[int],
  mesh_vert: wp.array[wp.vec3],
  mesh_graph: wp.array[int],
  mesh_polynum: wp.array[int],
  mesh_polyadr: wp.array[int],
  mesh_polynormal: wp.array[wp.vec3],
  mesh_polyvertadr: wp.array[int],
  mesh_polyvertnum: wp.array[int],
  mesh_polyvert: wp.array[int],
  mesh_polymapadr: wp.array[int],
  mesh_polymapnum: wp.array[int],
  mesh_polymap: wp.array[int],
  geom_xpos_in: wp.array[wp.vec3],
  geom_xmat_in: wp.array[wp.mat33],
  geoms: wp.vec2i,
  worldid: int,
) -> tuple[Geom, Geom]: ...
@wp.func
def write_contact(
  naconmax_in: int,
  id_: int,
  dist_in: float,
  pos_in: wp.vec3,
  frame_in: wp.mat33,
  margin_in: float,
  gap_in: float,
  condim_in: int,
  friction_in: vec5,
  solref_in: wp.vec2,
  solreffriction_in: wp.vec2,
  solimp_in: vec5,
  adhesion_in: float,
  geoms_in: wp.vec2i,
  pairid_in: wp.vec2i,
  worldid_in: int,
  contact_dist_out: wp.array[float],
  contact_pos_out: wp.array[wp.vec3],
  contact_frame_out: wp.array[wp.mat33],
  contact_includemargin_out: wp.array[float],
  contact_friction_out: wp.array[vec5],
  contact_solref_out: wp.array[wp.vec2],
  contact_solreffriction_out: wp.array[wp.vec2],
  contact_solimp_out: wp.array[vec5],
  contact_dim_out: wp.array[int],
  contact_geom_out: wp.array[wp.vec2i],
  contact_efc_address_out: wp.array[int],
  contact_worldid_out: wp.array[int],
  contact_type_out: wp.array[int],
  contact_geomcollisionid_out: wp.array[int],
  contact_adhesion_out: wp.array[float],
  nacon_out: wp.array[int],
) -> int: ...
@wp.func
def contact_margin_gap(
  geom_margin: wp.array[float],
  geom_gap: wp.array[float],
  pair_margin: wp.array[float],
  pair_gap: wp.array[float],
  geoms: wp.vec2i,
  pairid: int,
  worldid: int,
) -> tuple[float, float]: ...
@wp.func
def contact_material_params(
  geom_condim: wp.array[int],
  geom_priority: wp.array[int],
  geom_solmix: wp.array[float],
  geom_solref: wp.array[wp.vec2],
  geom_solimp: wp.array[vec5],
  geom_friction: wp.array[wp.vec3],
  geom_adhesion: wp.array[float],
  pair_dim: wp.array[int],
  pair_solref: wp.array[wp.vec2],
  pair_solreffriction: wp.array[wp.vec2],
  pair_solimp: wp.array[vec5],
  pair_adhesion: wp.array[float],
  pair_friction: wp.array[vec5],
  geoms: wp.vec2i,
  pairid: int,
  worldid: int,
) -> tuple[int, vec5, wp.vec2, wp.vec2, vec5, float]: ...
@wp.func
def contact_params(
  geom_condim: wp.array[int],
  geom_priority: wp.array[int],
  geom_solmix: wp.array[float],
  geom_solref: wp.array[wp.vec2],
  geom_solimp: wp.array[vec5],
  geom_friction: wp.array[wp.vec3],
  geom_margin: wp.array[float],
  geom_gap: wp.array[float],
  geom_adhesion: wp.array[float],
  pair_dim: wp.array[int],
  pair_solref: wp.array[wp.vec2],
  pair_solreffriction: wp.array[wp.vec2],
  pair_solimp: wp.array[vec5],
  pair_margin: wp.array[float],
  pair_gap: wp.array[float],
  pair_adhesion: wp.array[float],
  pair_friction: wp.array[vec5],
  collision_pair_in: wp.array[wp.vec2i],
  collision_pairid_in: wp.array[wp.vec2i],
  cid: int,
  worldid: int,
) -> tuple[
  wp.vec2i, float, float, int, vec5, wp.vec2, wp.vec2, vec5, float
]: ...

@dataclasses.dataclass
class CollisionContext:
  """Collision driver intermediate arrays."""

  collision_pair: wp.array[wp.vec2i]
  collision_pairid: wp.array[wp.vec2i]
  collision_worldid: wp.array[int]

@wp.func
def sap_binary_search(
  values: wp.array[_T], value: _T, lower: int, upper: int
) -> int: ...

sap_range: wp.Kernel

def create_collision_context(naconmax: int) -> CollisionContext: ...
