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
"""Support functions: Jacobians, mass matrix products and state access."""

from typing import Literal, overload

import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src.types import (
  Data,
  Model,
  vec5,
  vec10,
)

MAX_INTERP_BODIES: int

class vec_interp_body(wp.Vector[int, Literal[64]]): ...
class vec_interp_weight(wp.Vector[float, Literal[64]]): ...

@wp.func
def next_act(
  opt_timestep: float,
  actuator_dyntype: int,
  actuator_dynprm: vec10,
  actuator_actrange: wp.vec2,
  act_in: float,
  act_dot_in: float,
  act_dot_scale: float,
  clamp: bool,
) -> float: ...
@wp.func
def mat33_to_quat_polar(F: wp.mat33) -> wp.quat: ...
@wp.func
def compute_interp_cell_quat(
  flex_node0: wp.array[wp.vec3],
  flexnode_xpos_in: wp.array[wp.vec3],
  order: int,
  ci: int,
  cj: int,
  ck: int,
  cy: int,
  cz: int,
  ny_g: int,
  nz_g: int,
  nstart: int,
  worldid: int,
) -> wp.quat: ...
@wp.func
def rod_parallel_transport(
  t_from: wp.vec3, t_to: wp.vec3, u_in: wp.vec3
) -> tuple[bool, wp.vec3, wp.vec3]: ...
@wp.func
def rod_has_twist(flex_rod_has_twist: wp.array[bool], f: int) -> bool: ...
@wp.func
def rod_box_size(
  flex_rod_is_box: wp.array[bool], flex_rod_box_size: wp.array[wp.vec2], f: int
) -> tuple[bool, wp.vec2]: ...
@wp.func
def rod_clamped_ends(
  flex_rod_root_clamped: wp.array[bool],
  flex_rod_tail_clamped: wp.array[bool],
  f: int,
) -> tuple[bool, bool]:
  """Returns (root_clamped, tail_clamped) for 1D DER rod flex f."""

@wp.func
def rod_chain_twist_jnt(
  flex_rod_edgeadr: wp.array[int],
  flex_rod_edgenum: wp.array[int],
  flex_rodedge_jnt: wp.array[int],
  f: int,
  c: int,
) -> int:
  """Twist hinge joint ID of chain edge c of 1D DER rod f, or -1 if none."""

@wp.func
def rod_chain_twist_dof(
  jnt_dofadr: wp.array[int],
  flex_rod_edgeadr: wp.array[int],
  flex_rod_edgenum: wp.array[int],
  flex_rodedge_jnt: wp.array[int],
  f: int,
  c: int,
) -> int: ...
@wp.func
def rod_edge_twist_jnt(
  flex_elemadr: wp.array[int],
  flex_elemnum: wp.array[int],
  flex_rod_edgeadr: wp.array[int],
  flex_rod_edgenum: wp.array[int],
  flex_rodedge_jnt: wp.array[int],
  flex_elem_rodedge: wp.array[wp.vec2i],
  f: int,
  elem: int,
) -> tuple[int, float]: ...
@wp.func
def rod_vert_elem(
  flex_vertadr: wp.array[int],
  flex_vertnum: wp.array[int],
  flex_vert_rodelem: wp.array[int],
  f: int,
  v: int,
) -> int: ...
@wp.func
def rod_edge_state(
  plastic_in: wp.array[float], worldid: int, padr: int, c: int
) -> tuple[bool, wp.vec3, wp.vec3]: ...
@wp.func
def rod_edge_director(
  flex_vertadr: wp.array[int],
  flex_plasticadr: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_rod_root_clamped: wp.array[bool],
  flex_rod_tail_clamped: wp.array[bool],
  flex_rod_edgeadr: wp.array[int],
  flex_rod_edgenum: wp.array[int],
  flex_rodedge_vert: wp.array[wp.vec2i],
  plastic_in: wp.array[float],
  xmat_in: wp.array[wp.mat33],
  flexvert_xpos_in: wp.array[wp.vec3],
  flex_rod_frame0_loc_in: wp.array[wp.vec3],
  flex_rod_frame0_ref_in: wp.array[wp.vec3],
  flex_rod_tail_loc_in: wp.array[wp.vec3],
  worldid: int,
  f: int,
  c: int,
) -> tuple[bool, wp.vec3, wp.vec3, wp.vec3]: ...
@wp.func
def rod_chain_edge_director(
  flex_vertadr: wp.array[int],
  flex_plasticadr: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_rod_root_clamped: wp.array[bool],
  flex_rod_tail_clamped: wp.array[bool],
  flex_rod_edgeadr: wp.array[int],
  flex_rod_edgenum: wp.array[int],
  flex_rodedge_vert: wp.array[wp.vec2i],
  plastic_in: wp.array[float],
  xmat_in: wp.array[wp.mat33],
  flexvert_xpos_in: wp.array[wp.vec3],
  flex_rod_frame0_loc_in: wp.array[wp.vec3],
  flex_rod_frame0_ref_in: wp.array[wp.vec3],
  flex_rod_tail_loc_in: wp.array[wp.vec3],
  worldid: int,
  f: int,
  e: int,
  t_prev_edge: wp.vec3,
  u_prev_edge: wp.vec3,
) -> tuple[bool, bool, wp.vec3, wp.vec3, wp.vec3, wp.vec3, float]: ...
@wp.func
def rod_box_elem(
  qpos0: wp.array[float],
  jnt_qposadr: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_elemadr: wp.array[int],
  flex_elemnum: wp.array[int],
  flex_elemdataadr: wp.array[int],
  flex_plasticadr: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_elem: wp.array[int],
  flex_rod_is_box: wp.array[bool],
  flex_rod_root_clamped: wp.array[bool],
  flex_rod_tail_clamped: wp.array[bool],
  flex_rod_box_size: wp.array[wp.vec2],
  flex_rod_edgeadr: wp.array[int],
  flex_rod_edgenum: wp.array[int],
  flex_rodedge_vert: wp.array[wp.vec2i],
  flex_rodedge_jnt: wp.array[int],
  flex_elem_rodedge: wp.array[wp.vec2i],
  qpos_in: wp.array[float],
  plastic_in: wp.array[float],
  xmat_in: wp.array[wp.mat33],
  flexvert_xpos_in: wp.array[wp.vec3],
  flex_rod_frame0_loc_in: wp.array[wp.vec3],
  flex_rod_frame0_ref_in: wp.array[wp.vec3],
  flex_rod_tail_loc_in: wp.array[wp.vec3],
  worldid: int,
  f: int,
  e: int,
) -> tuple[bool, wp.vec3, wp.mat33, wp.vec3]: ...
@wp.func
def rod_contact_side_params(
  jnt_dofadr: wp.array[int],
  geom_type: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_elemadr: wp.array[int],
  flex_elemnum: wp.array[int],
  flex_elemdataadr: wp.array[int],
  flex_plasticadr: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_elem: wp.array[int],
  flex_rod_has_twist: wp.array[bool],
  flex_rod_edgeadr: wp.array[int],
  flex_rod_edgenum: wp.array[int],
  flex_rodedge_jnt: wp.array[int],
  flex_elem_rodedge: wp.array[wp.vec2i],
  flex_vert_rodelem: wp.array[int],
  plastic_in: wp.array[float],
  flexvert_xpos_in: wp.array[wp.vec3],
  worldid: int,
  side: int,
  geom: wp.vec2i,
  flex: wp.vec2i,
  elem: wp.vec2i,
  vert: wp.vec2i,
  con_dist: float,
  con_pos: wp.vec3,
  con_normal: wp.vec3,
) -> tuple[
  int,
  wp.vec3,
  wp.vec3,
  bool,
  int,
  int,
  wp.vec3,
  wp.vec3,
  float,
  float,
  float,
  wp.vec3,
  wp.vec3,
  wp.vec3,
  wp.vec3,
]: ...
@wp.func
def rod_contact_dof_delta(
  dof_axis: wp.spatial_vectorf,
  xk_offset: wp.vec3,
  con_minus_xk: wp.vec3,
  t: wp.vec3,
  dvec: wp.vec3,
  t_cross_r: wp.vec3,
  g_psi: wp.vec3,
  sgn_wk: float,
  sgn_esgn: float,
  inv_le: float,
) -> tuple[wp.vec3, wp.vec3]: ...
def mul_m_kernel(check_skip: bool) -> wp.Kernel: ...
def mul_m_dense(nv: int, check_skip: bool) -> wp.Kernel: ...
def mul_m(
  m: Model,
  d: Data,
  res: wp.array[float],
  vec: wp.array[float],
  skip: wp.array[bool] | None = ...,
  M: wp.array[float] | None = ...,
) -> None:
  """Multiply vectors by inertia matrix; optionally skip per world."""

def apply_ft(
  m: Model,
  d: Data,
  ft: wp.array[wp.spatial_vectorf],
  qfrc: wp.array[float],
  flg_add: bool,
) -> None: ...
def xfrc_accumulate(m: Model, d: Data, qfrc: wp.array[float]) -> None: ...
@wp.func
def contact_force_fn(
  opt_cone: int,
  contact_frame_in: wp.array[wp.mat33],
  contact_friction_in: wp.array[vec5],
  contact_dim_in: wp.array[int],
  contact_efc_address_in: wp.array[int],
  contact_adhesion_in: wp.array[float],
  efc_force_in: wp.array[float],
  njmax_in: int,
  nacon_in: wp.array[int],
  worldid: int,
  contact_id: int,
  to_world_frame: bool,
) -> wp.spatial_vectorf:
  """Extract 6D force:torque for one contact, in contact frame by default."""

contact_force_kernel: wp.Kernel

def contact_force(
  m: Model,
  d: Data,
  contact_ids: wp.array[int],
  to_world_frame: bool,
  force: wp.array[wp.spatial_vectorf],
) -> None:
  """Compute forces for contacts in Data."""

@overload
def transform_force(
  force: wp.vec3, torque: wp.vec3, offset: wp.vec3
) -> wp.spatial_vectorf: ...
@overload
def transform_force(
  frc: wp.spatial_vectorf, offset: wp.vec3
) -> wp.spatial_vectorf: ...
@wp.func
def jac_dof(
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  dof_bodyid: wp.array[int],
  body_isdofancestor: wp.array[int],
  subtree_com_in: wp.array[wp.vec3],
  cdof_in: wp.array[wp.spatial_vectorf],
  point: wp.vec3,
  bodyid: int,
  dofid: int,
  worldid: int,
) -> tuple[wp.vec3, wp.vec3]: ...
def jac(
  m: Model,
  d: Data,
  jacp: wp.array[float] | None,
  jacr: wp.array[float] | None,
  point: wp.array[wp.vec3],
  body: wp.array[int],
) -> None:
  """Compute translational and rotational Jacobian for point on body."""

@wp.func
def jac_dot_dof(
  body_parentid: wp.array[int],
  body_rootid: wp.array[int],
  jnt_type: wp.array[int],
  jnt_dofadr: wp.array[int],
  dof_bodyid: wp.array[int],
  dof_jntid: wp.array[int],
  body_isdofancestor: wp.array[int],
  subtree_com_in: wp.array[wp.vec3],
  cdof_in: wp.array[wp.spatial_vectorf],
  cvel_in: wp.array[wp.spatial_vectorf],
  cdof_dot_in: wp.array[wp.spatial_vectorf],
  point: wp.vec3,
  bodyid: int,
  dofid: int,
  worldid: int,
) -> tuple[wp.vec3, wp.vec3]: ...
def get_state(
  m: Model,
  d: Data,
  state: wp.array[float],
  sig: int,
  active: wp.array[bool] | None = ...,
) -> None: ...
def set_state(
  m: Model,
  d: Data,
  state: wp.array[float],
  sig: int,
  active: wp.array[bool] | None = ...,
) -> None: ...
@wp.func
def eval_basis_trilinear(local: wp.vec3, node_idx: int) -> float:
  """Evaluate trilinear basis function for node_idx at local coords [0,1]^3."""

@wp.func
def add_interp_weight(
  nb: int, body: vec_interp_body, weight: vec_interp_weight, b: int, w: float
) -> tuple[int, vec_interp_body, vec_interp_weight]:
  """Appends weight to body id, merging duplicates by summing weights."""

@wp.func
def add_interp_point_weights(
  body_weldid: wp.array[int],
  flex_nodebodyid: wp.array[int],
  coord: wp.vec3,
  scale: float,
  cn: wp.vec3i,
  nstart: int,
  is_shell: bool,
  use_weld: bool,
  local_nb: int,
  local_bodies: vec_interp_body,
  local_weights: vec_interp_weight,
) -> tuple[int, vec_interp_body, vec_interp_weight]: ...
@wp.func
def add_flex_interp_weights(
  body_weldid: wp.array[int],
  flex_dim: wp.array[int],
  flex_interp: wp.array[int],
  flex_cellnum: wp.array[wp.vec3i],
  flex_nodeadr: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_elemdataadr: wp.array[int],
  flex_nodebodyid: wp.array[int],
  flex_elem: wp.array[int],
  flex_vert0: wp.array[wp.vec3],
  flexvert_xpos_in: wp.array[wp.vec3],
  f: int,
  elem_id: int,
  vert_id: int,
  con_pos: wp.vec3,
  worldid: int,
  scale: float,
  use_weld: bool,
  local_nb: int,
  local_bodies: vec_interp_body,
  local_weights: vec_interp_weight,
) -> tuple[int, vec_interp_body, vec_interp_weight]: ...
@wp.func
def get_face_metadata(
  cellnum_x: int,
  cellnum_y: int,
  cellnum_z: int,
  face_elem_idx: int,
  order_abs: int,
) -> tuple[int, int, int, int, int, int]: ...
@wp.func
def gather_face_node_index_fast(
  normal_axis: int,
  g_fixed: int,
  q0: int,
  q1: int,
  ny_g: int,
  nz_g: int,
  local_idx: int,
  order_abs: int,
) -> int: ...
@wp.func
def gather_face_node_index(
  cellnum_x: int,
  cellnum_y: int,
  cellnum_z: int,
  face_elem_idx: int,
  local_idx: int,
  order_abs: int,
) -> int: ...
@wp.func
def compute_interp_face_quat(
  flex_node0: wp.array[wp.vec3],
  flexnode_xpos_in: wp.array[wp.vec3],
  cellnum_x: int,
  cellnum_y: int,
  cellnum_z: int,
  face_elem_idx: int,
  nstart: int,
  order_abs: int,
  worldid: int,
) -> wp.quat: ...
@wp.func
def flex_phi(s: float, i: int, order: int) -> float: ...
@wp.func
def flex_dphi(s: float, i: int, order: int) -> float: ...
@wp.func
def dphi2D(
  s0: float, l0: int, s1: float, l1: int, order: int, direction: int
) -> float: ...
@wp.func
def flex_elem_weights(
  flex_dim: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_elemdataadr: wp.array[int],
  flex_elem: wp.array[int],
  flexvert_xpos_in: wp.array[wp.vec3],
  flexid: int,
  elemid: int,
  pos: wp.vec3,
  worldid: int,
) -> wp.vec4: ...
@wp.func
def contact_bodies_and_weights(
  jnt_bodyid: wp.array[int],
  geom_bodyid: wp.array[int],
  flex_dim: wp.array[int],
  flex_interp: wp.array[int],
  flex_cellnum: wp.array[wp.vec3i],
  flex_nodeadr: wp.array[int],
  flex_vertadr: wp.array[int],
  flex_elemadr: wp.array[int],
  flex_elemnum: wp.array[int],
  flex_elemdataadr: wp.array[int],
  flex_nodebodyid: wp.array[int],
  flex_vertbodyid: wp.array[int],
  flex_elem: wp.array[int],
  flex_vert0: wp.array[wp.vec3],
  flex_rod_has_twist: wp.array[bool],
  flex_rod_edgeadr: wp.array[int],
  flex_rod_edgenum: wp.array[int],
  flex_rodedge_jnt: wp.array[int],
  flex_elem_rodedge: wp.array[wp.vec2i],
  flex_vert_rodelem: wp.array[int],
  flexvert_xpos_in: wp.array[wp.vec3],
  conid: int,
  side: int,
  geom: wp.vec2i,
  flex: wp.vec2i,
  elem: wp.vec2i,
  vert: wp.vec2i,
  con_pos: wp.vec3,
  worldid: int,
) -> tuple[wp.vec4i, wp.vec4]: ...
