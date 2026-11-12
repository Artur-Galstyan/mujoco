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
"""MJX Warp types."""

import dataclasses
from collections.abc import Iterable
from typing import TypeAlias

import jax
import numpy as np
import warp as wp
from jax import tree_util
from numpy import typing as npt
from typing_extensions import Self

from mujoco.mjx._src import dataclasses as mjx_dataclasses
from mujoco.mjx.third_party.mujoco_warp._src.types import Callback as Callback

GraphMode: TypeAlias = wp.JaxCallableGraphMode
PyTreeNode = mjx_dataclasses.PyTreeNode

@dataclasses.dataclass(frozen=True)
@tree_util.register_pytree_node_class
class TileSet:
  """Tiling configuration for decomposable block diagonal matrix."""

  adr: npt.NDArray[np.int32]
  size: int
  elemid: npt.NDArray[np.int32] = ...
  def __eq__(self, other: object) -> bool: ...
  def __hash__(self) -> int: ...
  def tree_flatten(
    self,
  ) -> tuple[list[npt.NDArray[np.int32] | int], None]: ...
  @classmethod
  def tree_unflatten(
    cls, aux_data: None, children: Iterable[npt.NDArray[np.int32] | int]
  ) -> Self: ...

@dataclasses.dataclass(frozen=True)
@tree_util.register_pytree_node_class
class BlockDim:
  """Block dimension 'block_dim' settings for wp.launch_tiled."""

  segmented_sort: int = ...
  nxn_broadphase: int = ...
  convex_ccd: int = ...
  island_dsu: int = ...
  ray: int = ...
  contact_sort: int = ...
  energy_vel_kinetic: int = ...
  cholesky_factorize: int = ...
  cholesky_factorize_solve: int = ...
  cholesky_solve: int = ...
  small_cholesky: int = ...
  solve_LD_sparse_fused: int = ...
  update_gradient_cholesky: int = ...
  update_gradient_cholesky_blocked: int = ...
  update_gradient_JTDAJ_sparse: int = ...
  update_gradient_JTDAJ_dense: int = ...
  linesearch_iterative: int = ...
  update_gradient_grad: int = ...
  solve_beta_accumulate: int = ...
  solve_search_update_cg: int = ...
  solve_init_search_cg: int = ...
  contact_jac_tiled: int = ...
  eff_pcg: int = ...
  render: int = ...
  def tree_flatten(self) -> tuple[list[int], None]: ...
  @classmethod
  def tree_unflatten(cls, aux_data: None, children: Iterable[int]) -> Self: ...

class OptionWarp(PyTreeNode):
  """Derived fields from Option."""

  broadphase: int
  broadphase_filter: int
  ccd_iterations: int
  ccd_tolerance: jax.Array
  contact_sensor_maxmatch: int
  deterministic: int
  graph_conditional: bool
  graph_mode: GraphMode
  impratio_invsqrt: jax.Array
  noslip_iterations: int
  noslip_tolerance: jax.Array
  run_collision_detection: bool
  run_qacc_smooth: bool
  run_rne_postconstraint: bool
  sdf_initpoints: int
  sdf_iterations: int
  sleep_tolerance: jax.Array
  warn_overflow: int

class ModelWarp(PyTreeNode):
  """Derived fields from Model."""

  D_colind: npt.NDArray[np.int32]
  D_diag: npt.NDArray[np.int32]
  D_rowadr: npt.NDArray[np.int32]
  D_rownnz: npt.NDArray[np.int32]
  M_elemid: npt.NDArray[np.int32]
  M_fullm_i: npt.NDArray[np.int32]
  M_fullm_j: npt.NDArray[np.int32]
  M_hinit_i: npt.NDArray[np.int32]
  M_mulm_col: npt.NDArray[np.int32]
  M_mulm_madr: npt.NDArray[np.int32]
  M_mulm_rowadr: npt.NDArray[np.int32]
  M_tiles: tuple[TileSet, ...]
  actuator_armature: jax.Array
  actuator_ctrladr: npt.NDArray[np.int32]
  actuator_ctrlnum: npt.NDArray[np.int32]
  actuator_ctrlspec: npt.NDArray[np.int32]
  actuator_damping: jax.Array
  actuator_dampingpoly: jax.Array
  actuator_delay: jax.Array
  actuator_history: jax.Array
  actuator_historyadr: jax.Array
  actuator_trntype_body_adr: npt.NDArray[np.int32]
  block_dim: BlockDim
  body_branch_start: npt.NDArray[np.int32]
  body_branches: npt.NDArray[np.int32]
  body_fluid_box_adr: npt.NDArray[np.int32]
  body_fluid_ellipsoid: npt.NDArray[np.bool_]
  body_fluid_ellipsoid_adr: npt.NDArray[np.int32]
  body_freeadr: npt.NDArray[np.int32]
  body_is_free: npt.NDArray[np.bool_]
  body_isdofancestor: npt.NDArray[np.int32]
  body_subtreenum: npt.NDArray[np.int32]
  body_tree: tuple[npt.NDArray[np.int32], ...]
  callback: Callback
  cam_projection: npt.NDArray[np.int32]
  collision_sensor_adr: npt.NDArray[np.int32]
  dof_length: npt.NDArray[np.float32]
  dof_tri_col: npt.NDArray[np.int32]
  dof_tri_row: npt.NDArray[np.int32]
  efm0_L: npt.NDArray[np.float32]
  efm0_L_colind: npt.NDArray[np.int32]
  efm0_L_rowadr: npt.NDArray[np.int32]
  efm0_L_rownnz: npt.NDArray[np.int32]
  efm0_active: bool
  efm0_dofid: npt.NDArray[np.int32]
  efm_K_colind: npt.NDArray[np.int32]
  efm_K_rowadr: npt.NDArray[np.int32]
  efm_K_rownnz: npt.NDArray[np.int32]
  efm_con_max_dof: int
  efm_dofid: npt.NDArray[np.int32]
  eq_connect_adr: npt.NDArray[np.int32]
  eq_flex_adr: npt.NDArray[np.int32]
  eq_flexstrain_adr: npt.NDArray[np.int32]
  eq_jnt_adr: npt.NDArray[np.int32]
  eq_ten_adr: npt.NDArray[np.int32]
  eq_wld_adr: npt.NDArray[np.int32]
  flex_activelayers: npt.NDArray[np.int32]
  flex_bend_interp_map: npt.NDArray[np.int32]
  flex_bending: npt.NDArray[np.float32]
  flex_bendingadr: npt.NDArray[np.int32]
  flex_cell_map: npt.NDArray[np.int32]
  flex_cellnum: npt.NDArray[np.int32]
  flex_centered: npt.NDArray[np.bool_]
  flex_conaffinity: npt.NDArray[np.int32]
  flex_condim: npt.NDArray[np.int32]
  flex_contype: npt.NDArray[np.int32]
  flex_damping: npt.NDArray[np.float32]
  flex_dim: npt.NDArray[np.int32]
  flex_edge: npt.NDArray[np.int32]
  flex_edgeadr: npt.NDArray[np.int32]
  flex_edgedamping: npt.NDArray[np.float32]
  flex_edgeequality: npt.NDArray[np.int32]
  flex_edgeflap: npt.NDArray[np.int32]
  flex_edgeflexid: npt.NDArray[np.int32]
  flex_edgenum: npt.NDArray[np.int32]
  flex_edgestiffness: npt.NDArray[np.float32]
  flex_efcnnz: npt.NDArray[np.int32]
  flex_efcnum: npt.NDArray[np.int32]
  flex_eff_possible: npt.NDArray[np.bool_]
  flex_elem: npt.NDArray[np.int32]
  flex_elem_rodedge: npt.NDArray[np.int32]
  flex_elemadr: npt.NDArray[np.int32]
  flex_elemdataadr: npt.NDArray[np.int32]
  flex_elemedge: npt.NDArray[np.int32]
  flex_elemedgeadr: npt.NDArray[np.int32]
  flex_elemflexid: npt.NDArray[np.int32]
  flex_elemlayer: npt.NDArray[np.int32]
  flex_elemnum: npt.NDArray[np.int32]
  flex_face: npt.NDArray[np.int32]
  flex_face_map: npt.NDArray[np.int32]
  flex_faceadr: npt.NDArray[np.int32]
  flex_friction: npt.NDArray[np.float32]
  flex_gap: npt.NDArray[np.float32]
  flex_interp_assemblable: bool
  flex_margin: npt.NDArray[np.float32]
  flex_node: npt.NDArray[np.float32]
  flex_passive: npt.NDArray[np.int32]
  flex_plasticadr: npt.NDArray[np.int32]
  flex_plasticmode: npt.NDArray[np.int32]
  flex_priority: npt.NDArray[np.int32]
  flex_radius: npt.NDArray[np.float32]
  flex_rigid: npt.NDArray[np.bool_]
  flex_rod_box_size: npt.NDArray[np.float32]
  flex_rod_edgeadr: npt.NDArray[np.int32]
  flex_rod_edgenum: npt.NDArray[np.int32]
  flex_rod_has_twist: npt.NDArray[np.bool_]
  flex_rod_hingeadr: npt.NDArray[np.int32]
  flex_rod_is_box: npt.NDArray[np.bool_]
  flex_rod_is_chain: npt.NDArray[np.bool_]
  flex_rod_root_clamped: npt.NDArray[np.bool_]
  flex_rod_tail_clamped: npt.NDArray[np.bool_]
  flex_rodedge_jnt: npt.NDArray[np.int32]
  flex_rodedge_vert: npt.NDArray[np.int32]
  flex_rodhinge_map: npt.NDArray[np.int32]
  flex_selfcollide: npt.NDArray[np.int32]
  flex_shell: npt.NDArray[np.int32]
  flex_shelladr: npt.NDArray[np.int32]
  flex_shelldataadr: npt.NDArray[np.int32]
  flex_shellflexid: npt.NDArray[np.int32]
  flex_shellnum: npt.NDArray[np.int32]
  flex_simple: npt.NDArray[np.bool_]
  flex_size: npt.NDArray[np.float32]
  flex_solimp: npt.NDArray[np.float32]
  flex_solmix: npt.NDArray[np.float32]
  flex_solref: npt.NDArray[np.float32]
  flex_stiffness: npt.NDArray[np.float32]
  flex_stiffnessadr: npt.NDArray[np.int32]
  flex_vert: npt.NDArray[np.float32]
  flex_vert_rodelem: npt.NDArray[np.int32]
  flex_vertbodyid: npt.NDArray[np.int32]
  flex_vertcell: npt.NDArray[np.int32]
  flex_vertflexid: npt.NDArray[np.int32]
  flex_yield: npt.NDArray[np.float32]
  flexedge_J_colind: npt.NDArray[np.int32]
  flexedge_J_rowadr: npt.NDArray[np.int32]
  flexedge_J_rownnz: npt.NDArray[np.int32]
  flexedge_efcadr: npt.NDArray[np.int32]
  flexedge_efcnnzadr: npt.NDArray[np.int32]
  flexedge_invweight0: npt.NDArray[np.float32]
  flexedge_length0: npt.NDArray[np.float32]
  flexedge_rigid: npt.NDArray[np.bool_]
  flexstrain_J_colind: npt.NDArray[np.int32]
  flexstrain_J_rowadr: npt.NDArray[np.int32]
  flexstrain_J_rownnz: npt.NDArray[np.int32]
  flg_adhesion: bool
  geom_adhesion: jax.Array
  geom_pair_type_count: tuple[int, ...]
  geom_plugin_index: npt.NDArray[np.int32]
  geom_surfacevel: jax.Array
  has_1d_flex: bool
  has_2d_flex: bool
  has_3d_flex: bool
  has_actuator_dcmotor: bool
  has_actuator_wrap: bool
  has_efm_actuator: bool
  has_ellipsoid_geom: bool
  has_flex_edge_stiff_damp: bool
  has_flex_passive: bool
  has_flex_selfcollide: bool
  has_flex_snh: bool
  has_fluid: bool
  has_non_simple_flex: bool
  has_plane_geom: bool
  has_sdf_geom: bool
  has_tendon_damping: bool
  has_tendon_stiffness: bool
  has_unsupported_flex_interp: bool
  is_sparse: bool
  jnt_actuatorid: npt.NDArray[np.int32]
  jnt_limited_ball_adr: npt.NDArray[np.int32]
  jnt_limited_slide_hinge_adr: npt.NDArray[np.int32]
  jnt_plasticadr: npt.NDArray[np.int32]
  jnt_springyield: jax.Array
  light_bodyid: npt.NDArray[np.int32]
  light_targetbodyid: npt.NDArray[np.int32]
  mapD2M: npt.NDArray[np.int32]
  mapM2D: npt.NDArray[np.int32]
  mapM2M: npt.NDArray[np.int32]
  mat_texrepeat: jax.Array
  mat_texuniform: jax.Array
  max_flex_dim: int
  max_ten_J_rownnz: int
  mesh_polyadr: npt.NDArray[np.int32]
  mesh_polymap: npt.NDArray[np.int32]
  mesh_polymapadr: npt.NDArray[np.int32]
  mesh_polymapnum: npt.NDArray[np.int32]
  mesh_polynormal: npt.NDArray[np.float32]
  mesh_polynum: npt.NDArray[np.int32]
  mesh_polyvert: npt.NDArray[np.int32]
  mesh_polyvertadr: npt.NDArray[np.int32]
  mesh_polyvertnum: npt.NDArray[np.int32]
  mocap_bodyid: npt.NDArray[np.int32]
  nJfe: int
  nJfs: int
  nacttrnbody: int
  nactuator: int
  nbranch: int
  nefm0L: int
  nefm0dof: int
  nefmK: int
  nefmdof: int
  neq_flexstrain: int
  nflexbend_interp: int
  nflexbending: int
  nflexedge: int
  nflexelem: int
  nflexelemdata: int
  nflexelemedge: int
  nflexface: int
  nflexintcell: int
  nflexnode: int
  nflexrodedge: int
  nflexrodhinge: int
  nflexshelldata: int
  nflexstiffness: int
  nflexvert: int
  nmaxcondim: int
  nmaxpyramid: int
  nmeshdegmax: int
  noct: int
  nplugin: int
  npolygonmax: int
  nrangefinder: int
  nsensorcollision: int
  nsensortaxel: int
  ntactileweld: int
  ntree: int
  nv_pad: int
  nxn_geom_pair: npt.NDArray[np.int32]
  nxn_geom_pair_filtered: npt.NDArray[np.int32]
  nxn_pairid: npt.NDArray[np.int32]
  nxn_pairid_filtered: npt.NDArray[np.int32]
  oct_aabb: npt.NDArray[np.float32]
  oct_child: npt.NDArray[np.int32]
  oct_coeff: npt.NDArray[np.float32]
  pair_adhesion: jax.Array
  plugin: npt.NDArray[np.int32]
  plugin_attr: npt.NDArray[np.float32]
  qD_fullm_i: npt.NDArray[np.int32]
  qD_fullm_j: npt.NDArray[np.int32]
  qLD_all_updates: npt.NDArray[np.int32]
  qLD_block_adr: npt.NDArray[np.int32]
  qLD_block_total: int
  qLD_level_offsets: npt.NDArray[np.int32]
  qLD_updates: tuple[npt.NDArray[np.int32], ...]
  rangefinder_sensor_adr: npt.NDArray[np.int32]
  sensor_acc_adr: npt.NDArray[np.int32]
  sensor_collision_start_adr: npt.NDArray[np.int32]
  sensor_contact_acc_adr: npt.NDArray[np.int32]
  sensor_contact_pos_adr: npt.NDArray[np.int32]
  sensor_contact_vel_adr: npt.NDArray[np.int32]
  sensor_delay: jax.Array
  sensor_e_kinetic: bool
  sensor_e_potential: bool
  sensor_history: jax.Array
  sensor_historyadr: jax.Array
  sensor_interval: jax.Array
  sensor_limitfrc_adr: npt.NDArray[np.int32]
  sensor_limitpos_adr: npt.NDArray[np.int32]
  sensor_limitvel_adr: npt.NDArray[np.int32]
  sensor_pos_adr: npt.NDArray[np.int32]
  sensor_rangefinder_adr: npt.NDArray[np.int32]
  sensor_rangefinder_bodyid: npt.NDArray[np.int32]
  sensor_rne_postconstraint: bool
  sensor_stage_acc_adr: npt.NDArray[np.int32]
  sensor_stage_pos_adr: npt.NDArray[np.int32]
  sensor_stage_vel_adr: npt.NDArray[np.int32]
  sensor_subtree_vel: bool
  sensor_tendonactfrc_adr: npt.NDArray[np.int32]
  sensor_touch_adr: npt.NDArray[np.int32]
  sensor_vel_adr: npt.NDArray[np.int32]
  taxel_sensorid: npt.NDArray[np.int32]
  taxel_vertadr: npt.NDArray[np.int32]
  ten_J_colind: npt.NDArray[np.int32]
  ten_J_rowadr: npt.NDArray[np.int32]
  ten_J_rownnz: npt.NDArray[np.int32]
  ten_wrapadr_site: npt.NDArray[np.int32]
  ten_wrapnum_site: npt.NDArray[np.int32]
  tendon_actuatorid: npt.NDArray[np.int32]
  tendon_geom_adr: npt.NDArray[np.int32]
  tendon_jnt_adr: npt.NDArray[np.int32]
  tendon_limited_adr: npt.NDArray[np.int32]
  tendon_plasticadr: npt.NDArray[np.int32]
  tendon_site_pair_adr: npt.NDArray[np.int32]
  tendon_springyield: jax.Array
  tree_bodynum: npt.NDArray[np.int32]
  tree_dofadr: npt.NDArray[np.int32]
  tree_dofnum: npt.NDArray[np.int32]
  tree_sleep_policy: npt.NDArray[np.int32]
  weld_tactile_id: npt.NDArray[np.int32]
  wrap_geom_adr: npt.NDArray[np.int32]
  wrap_jnt_adr: npt.NDArray[np.int32]
  wrap_pulley_scale: npt.NDArray[np.float32]
  wrap_site_adr: npt.NDArray[np.int32]
  wrap_site_pair_adr: npt.NDArray[np.int32]

class DataWarp(PyTreeNode):
  """Derived fields from Data."""

  M: jax.Array
  actuator_moment: jax.Array
  actuator_velocity: jax.Array
  body_awake: jax.Array
  body_awake_ind: jax.Array
  cJ: jax.Array
  cM: jax.Array
  cMa: jax.Array
  cacc: jax.Array
  cdof_dof: jax.Array
  cdof_tri_col: jax.Array
  cdof_tri_row: jax.Array
  cfrc_ext: jax.Array
  cfrc_int: jax.Array
  cinert: jax.Array
  contact__adhesion: jax.Array
  contact__dim: jax.Array
  contact__dist: jax.Array
  contact__efc_address: jax.Array
  contact__elem: jax.Array
  contact__flex: jax.Array
  contact__frame: jax.Array
  contact__friction: jax.Array
  contact__geom: jax.Array
  contact__geomcollisionid: jax.Array
  contact__includemargin: jax.Array
  contact__pos: jax.Array
  contact__solimp: jax.Array
  contact__solref: jax.Array
  contact__solreffriction: jax.Array
  contact__surfacevel: jax.Array
  contact__type: jax.Array
  contact__vert: jax.Array
  contact__worldid: jax.Array
  cqLD: jax.Array
  cqacc: jax.Array
  cqacc_smooth: jax.Array
  cqacc_warmstart: jax.Array
  cqfrc_constraint: jax.Array
  cqfrc_smooth: jax.Array
  crb: jax.Array
  crhs: jax.Array
  cx: jax.Array
  dof_awake_ind: jax.Array
  dof_cdof: jax.Array
  dof_frictionadr: jax.Array
  dof_island: jax.Array
  dof_islandid: jax.Array
  efc__D: jax.Array
  efc__J: jax.Array
  efc__J_colind: jax.Array
  efc__J_rowadr: jax.Array
  efc__J_rownnz: jax.Array
  efc__Jqvel: jax.Array
  efc__Ma: jax.Array
  efc__aref: jax.Array
  efc__force: jax.Array
  efc__frictionloss: jax.Array
  efc__id: jax.Array
  efc__island: jax.Array
  efc__jtdaj_adr: jax.Array
  efc__jtdaj_nblock: jax.Array
  efc__jtdaj_nrow: jax.Array
  efc__margin: jax.Array
  efc__pos: jax.Array
  efc__state: jax.Array
  efc__type: jax.Array
  efc__vel: jax.Array
  efc_islandid: jax.Array
  efm_K_val: jax.Array
  efm_as: jax.Array
  efm_c: jax.Array
  efm_ca: jax.Array
  efm_diag: jax.Array
  efm_fluid: jax.Array
  efm_ts: jax.Array
  energy: jax.Array
  face_quat: jax.Array
  face_xpos: jax.Array
  flex_aabb_max: jax.Array
  flex_aabb_min: jax.Array
  flex_hessian_valid: jax.Array
  flex_rod_chain_ok: jax.Array
  flex_rod_chain_valid: jax.Array
  flex_rod_end_torque: jax.Array
  flex_rod_frame0_loc: jax.Array
  flex_rod_frame0_ref: jax.Array
  flex_rod_ref_ok: jax.Array
  flex_rod_tail_loc: jax.Array
  flex_rodedge_frc: jax.Array
  flex_rodedge_grad: jax.Array
  flex_rodedge_ktheta: jax.Array
  flex_rodedge_tridiag: jax.Array
  flex_rodhinge_Wd: jax.Array
  flex_rodhinge_Wk: jax.Array
  flex_rodhinge_g: jax.Array
  flex_rodhinge_om0: jax.Array
  flexedge_J: jax.Array
  flexedge_hessian: jax.Array
  flexedge_length: jax.Array
  flexedge_velocity: jax.Array
  flexnode_xpos: jax.Array
  flexvert_hessian: jax.Array
  flexvert_xpos: jax.Array
  island_dofadr: jax.Array
  island_idofadr: jax.Array
  island_iefcadr: jax.Array
  island_ne: jax.Array
  island_nefc: jax.Array
  island_nf: jax.Array
  island_nv: jax.Array
  light_xdir: jax.Array
  light_xpos: jax.Array
  map_dof2idof: jax.Array
  map_efc2iefc: jax.Array
  map_idof2dof: jax.Array
  map_iefc2efc: jax.Array
  moment_colind: jax.Array
  moment_rowadr: jax.Array
  moment_rownnz: jax.Array
  naccdmax: int
  nacon: jax.Array
  naconmax: int
  nbody_awake: jax.Array
  ncdof: jax.Array
  ncollision: jax.Array
  ne: jax.Array
  nefc: jax.Array
  nf: jax.Array
  nidof: jax.Array
  nisland: jax.Array
  njmax: int
  njmax_nnz: int
  njmax_pad: int
  nl: jax.Array
  ntree_awake: jax.Array
  nv_awake: jax.Array
  nvmax: int
  nvmax_pad: int
  nworld: int
  overflow: jax.Array
  qH: jax.Array
  qHDiagInv: jax.Array
  qHLD: jax.Array
  qLD: jax.Array
  qLDiagInv: jax.Array
  qLU: jax.Array
  qfrc_adhesion: jax.Array
  qfrc_damper: jax.Array
  qfrc_spring: jax.Array
  solver_niter: jax.Array
  subtree_angmom: jax.Array
  subtree_linvel: jax.Array
  ten_J: jax.Array
  ten_velocity: jax.Array
  ten_wrapadr: jax.Array
  ten_wrapnum: jax.Array
  tree_asleep: jax.Array
  tree_awake: jax.Array
  tree_island: jax.Array
  wrap_obj: jax.Array
  wrap_xpos: jax.Array
  _jax_token: jax.Array = ...

  @property
  def shape(self) -> tuple[int, ...]: ...

DATA_NON_VMAP: set[str]
_NDIM: dict[str, dict[str, int]]
_BATCH_DIM: dict[str, dict[str, bool]]
