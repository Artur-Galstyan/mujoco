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
"""Base types used in MJX."""

import enum
from collections.abc import Iterable
from typing import Any, ClassVar, TypeAlias

import jax
import numpy as np
from numpy import typing as npt

import mujoco
from mujoco.mjx._src.dataclasses import PyTreeNode as PyTreeNode
from mujoco.mjx._src.support import BindData, BindModel
from mujoco.mjx.warp import types as mjxw_types

_Index: TypeAlias = (
  int
  | slice
  | jax.Array
  | npt.NDArray[np.integer]
  | npt.NDArray[np.bool_]
  | tuple[int | slice | jax.Array | npt.NDArray[np.generic] | None, ...]
)

class Impl(enum.Enum):
  """Implementation to use."""

  CPP = 'cpp'
  C = 'cpp'
  JAX = 'jax'
  WARP = 'warp'

class DisableBit(enum.IntFlag):
  """Disable default feature bitflags."""

  CONSTRAINT = 1
  EQUALITY = 2
  FRICTIONLOSS = 4
  LIMIT = 8
  CONTACT = 16
  SPRING = 32
  DAMPER = 64
  GRAVITY = 128
  CLAMPCTRL = 256
  WARMSTART = 512
  ACTUATION = 2048
  REFSAFE = 4096
  SENSOR = 8192
  EULERDAMP = 32768
  FILTERPARENT = 1024

class EnableBit(enum.IntFlag):
  """Enable optional feature bitflags."""

  INVDISCRETE = 8
  SLEEP = 16
  IPC = 64

class JointType(enum.IntEnum):
  """Type of degree of freedom."""

  FREE = 0
  BALL = 1
  SLIDE = 2
  HINGE = 3

  def dof_width(self) -> int: ...
  def qpos_width(self) -> int: ...

class IntegratorType(enum.IntEnum):
  """Integrator mode."""

  EULER = 0
  RK4 = 1
  IMPLICITFAST = 3
  DISCRETE = 4

class GeomType(enum.IntEnum):
  """Type of geometry."""

  PLANE = 0
  HFIELD = 1
  SPHERE = 2
  CAPSULE = 3
  ELLIPSOID = 4
  CYLINDER = 5
  BOX = 6
  MESH = 7

class ConvexMesh(PyTreeNode):
  """Geom properties for convex meshes."""

  vert: jax.Array
  face: jax.Array
  face_normal: jax.Array
  edge: jax.Array
  edge_face_normal: jax.Array

class ConeType(enum.IntEnum):
  """Type of friction cone."""

  PYRAMIDAL = 0
  ELLIPTIC = 1

class JacobianType(enum.IntEnum):
  """Type of constraint Jacobian."""

  DENSE = 0
  SPARSE = 1
  AUTO = 2

class SolverType(enum.IntEnum):
  """Constraint solver algorithm."""

  CG = 1
  NEWTON = 2

class EqType(enum.IntEnum):
  """Type of equality constraint."""

  CONNECT = 0
  WELD = 1
  JOINT = 2
  TENDON = 3

class TendonType(enum.IntEnum):
  """Type of tendon."""

  FIXED = 0
  SPATIAL = 1

class WrapType(enum.IntEnum):
  """Type of tendon wrap object."""

  JOINT = 1
  PULLEY = 2
  SITE = 3
  SPHERE = 4
  CYLINDER = 5

class TrnType(enum.IntEnum):
  """Type of actuator transmission."""

  JOINT = 0
  JOINTINPARENT = 1
  SITE = 4
  TENDON = 3

class DynType(enum.IntEnum):
  """Type of actuator dynamics."""

  NONE = 0
  INTEGRATOR = 1
  FILTER = 2
  FILTEREXACT = 3
  MUSCLE = 4

class GainType(enum.IntEnum):
  """Type of actuator gain."""

  FIXED = 0
  AFFINE = 1
  MUSCLE = 2

class BiasType(enum.IntEnum):
  """Type of actuator bias."""

  NONE = 0
  AFFINE = 1
  MUSCLE = 2

class ConstraintType(enum.IntEnum):
  """Type of constraint."""

  EQUALITY = 0
  FRICTION_DOF = 1
  FRICTION_TENDON = 2
  LIMIT_JOINT = 3
  LIMIT_TENDON = 4
  CONTACT_FRICTIONLESS = 5
  CONTACT_PYRAMIDAL = 6
  CONTACT_ELLIPTIC = 7

class CamLightType(enum.IntEnum):
  """Type of camera light."""

  FIXED = 0
  TRACK = 1
  TRACKCOM = 2
  TARGETBODY = 3
  TARGETBODYCOM = 4

class SensorType(enum.IntEnum):
  """Type of sensor."""

  MAGNETOMETER = 6
  CAMPROJECTION = 8
  RANGEFINDER = 7
  JOINTPOS = 9
  TENDONPOS = 11
  ACTUATORPOS = 13
  BALLQUAT = 18
  FRAMEPOS = 26
  FRAMEXAXIS = 28
  FRAMEYAXIS = 29
  FRAMEZAXIS = 30
  FRAMEQUAT = 27
  SUBTREECOM = 35
  CLOCK = 45
  VELOCIMETER = 2
  GYRO = 3
  JOINTVEL = 10
  TENDONVEL = 12
  ACTUATORVEL = 14
  BALLANGVEL = 19
  FRAMELINVEL = 31
  FRAMEANGVEL = 32
  SUBTREELINVEL = 36
  SUBTREEANGMOM = 37
  TOUCH = 0
  CONTACT = 42
  ACCELEROMETER = 1
  FORCE = 4
  TORQUE = 5
  ACTUATORFRC = 15
  JOINTACTFRC = 16
  TENDONACTFRC = 17
  FRAMELINACC = 33
  FRAMEANGACC = 34
  TACTILE = 46

class ObjType(PyTreeNode):
  """Type of object."""

  UNKNOWN: ClassVar[mujoco.mjtObj]
  BODY: ClassVar[mujoco.mjtObj]
  XBODY: ClassVar[mujoco.mjtObj]
  GEOM: ClassVar[mujoco.mjtObj]
  SITE: ClassVar[mujoco.mjtObj]
  CAMERA: ClassVar[mujoco.mjtObj]

class Statistic(PyTreeNode):
  """Model statistics (in qpos0)."""

  meaninertia: jax.Array
  meanmass: jax.Array
  meansize: jax.Array
  extent: jax.Array
  center: jax.Array

class OptionJAX(PyTreeNode):
  """JAX-specific option."""

  o_margin: jax.Array
  o_solref: jax.Array
  o_solimp: jax.Array
  o_friction: jax.Array
  disableactuator: int
  sdf_initpoints: int
  has_fluid_params: bool | np.bool_

class Option(PyTreeNode):
  """Physics options."""

  iterations: int
  ls_iterations: int
  tolerance: jax.Array
  ls_tolerance: jax.Array
  impratio: jax.Array
  gravity: jax.Array
  density: jax.Array
  viscosity: jax.Array
  magnetic: jax.Array
  wind: jax.Array
  jacobian: JacobianType
  cone: ConeType
  disableflags: DisableBit
  enableflags: int
  integrator: IntegratorType
  solver: SolverType
  timestep: jax.Array
  _impl: OptionJAX | mjxw_types.OptionWarp

class ModelCPP(PyTreeNode):
  """Minimal Model implementation holding only the pointer."""

  pointer_lo: jax.Array
  pointer_hi: jax.Array

class DataCPP(PyTreeNode):
  """Minimal Data implementation holding only the pointer."""

  pointer_lo: jax.Array
  pointer_hi: jax.Array

class ModelJAX(PyTreeNode):
  """JAX-specific model data."""

  dof_hasfrictionloss: npt.NDArray[np.bool_]
  geom_rbound_hfield: npt.NDArray[np.float64]
  mesh_convex: tuple[ConvexMesh, ...]
  tendon_hasfrictionloss: npt.NDArray[np.bool_]
  wrap_inside_maxiter: int
  wrap_inside_tolerance: float
  wrap_inside_z_init: float
  is_wrap_inside: npt.NDArray[np.bool_]

class Model(PyTreeNode):
  """Static model of the scene that remains unchanged with each physics step."""

  nq: int
  nv: int
  nu: int
  na: int
  nbody: int
  njnt: int
  ngeom: int
  nsite: int
  ncam: int
  nlight: int
  nflex: int
  nmesh: int
  nmeshvert: int
  nmeshnormal: int
  nmeshtexcoord: int
  nmeshface: int
  nmeshgraph: int
  nmeshpoly: int
  nmeshpolyvert: int
  nmeshpolymap: int
  nhfield: int
  nhfielddata: int
  ntex: int
  ntexdata: int
  nmat: int
  npair: int
  nexclude: int
  neq: int
  ntendon: int
  nwrap: int
  nsensor: int
  nnumeric: int
  ntuple: int
  nkey: int
  nmocap: int
  nM: int
  nB: int
  nC: int
  nD: int
  nJmom: int
  nJten: int
  ngravcomp: int
  flg_gravcomp: bool
  flg_surfacevel: bool
  nuserdata: int
  nsensordata: int
  npluginstate: int
  nhistory: int
  nplastic: int
  opt: Option
  stat: Statistic
  qpos0: jax.Array
  qpos_spring: jax.Array
  body_parentid: npt.NDArray[np.int32]
  body_mocapid: npt.NDArray[np.int32]
  body_rootid: npt.NDArray[np.int32]
  body_weldid: npt.NDArray[np.int32]
  body_jntnum: npt.NDArray[np.int32]
  body_jntadr: npt.NDArray[np.int32]
  body_sameframe: npt.NDArray[np.uint8]
  body_dofnum: npt.NDArray[np.int32]
  body_dofadr: npt.NDArray[np.int32]
  body_treeid: npt.NDArray[np.int32]
  body_geomnum: npt.NDArray[np.int32]
  body_geomadr: npt.NDArray[np.int32]
  body_simple: npt.NDArray[np.uint8]
  body_pos: jax.Array
  body_quat: jax.Array
  body_ipos: jax.Array
  body_iquat: jax.Array
  body_mass: jax.Array
  body_subtreemass: jax.Array
  body_inertia: jax.Array
  body_gravcomp: jax.Array
  body_margin: npt.NDArray[np.float64]
  body_contype: npt.NDArray[np.int32]
  body_conaffinity: npt.NDArray[np.int32]
  body_invweight0: jax.Array
  jnt_type: npt.NDArray[np.int32]
  jnt_qposadr: npt.NDArray[np.int32]
  jnt_dofadr: npt.NDArray[np.int32]
  jnt_bodyid: npt.NDArray[np.int32]
  jnt_limited: npt.NDArray[np.bool_]
  jnt_actfrclimited: npt.NDArray[np.bool_]
  jnt_actgravcomp: npt.NDArray[np.bool_]
  jnt_solref: jax.Array
  jnt_solimp: jax.Array
  jnt_pos: jax.Array
  jnt_axis: jax.Array
  jnt_stiffness: jax.Array
  jnt_stiffnesspoly: jax.Array
  jnt_range: jax.Array
  jnt_actfrcrange: jax.Array
  jnt_margin: jax.Array
  dof_bodyid: npt.NDArray[np.int32]
  dof_jntid: npt.NDArray[np.int32]
  dof_parentid: npt.NDArray[np.int32]
  dof_treeid: npt.NDArray[np.int32]
  dof_Madr: npt.NDArray[np.int32]
  dof_simplenum: npt.NDArray[np.int32]
  M_rowadr: npt.NDArray[np.int32]
  M_rownnz: npt.NDArray[np.int32]
  M_colind: npt.NDArray[np.int32]
  dof_solref: jax.Array
  dof_solimp: jax.Array
  dof_frictionloss: jax.Array
  dof_armature: jax.Array
  dof_damping: jax.Array
  dof_dampingpoly: jax.Array
  dof_invweight0: jax.Array
  dof_M0: jax.Array
  geom_type: npt.NDArray[np.int32]
  geom_contype: npt.NDArray[np.int32]
  geom_conaffinity: npt.NDArray[np.int32]
  geom_condim: npt.NDArray[np.int32]
  geom_bodyid: npt.NDArray[np.int32]
  geom_sameframe: npt.NDArray[np.uint8]
  geom_dataid: npt.NDArray[np.int32]
  geom_group: npt.NDArray[np.int32]
  geom_matid: jax.Array
  geom_priority: npt.NDArray[np.int32]
  geom_solmix: jax.Array
  geom_solref: jax.Array
  geom_solimp: jax.Array
  geom_size: jax.Array
  geom_aabb: jax.Array
  geom_rbound: jax.Array
  geom_pos: jax.Array
  geom_quat: jax.Array
  geom_friction: jax.Array
  geom_margin: jax.Array
  geom_gap: jax.Array
  geom_fluid: npt.NDArray[np.float64]
  geom_rgba: jax.Array
  site_type: npt.NDArray[np.int32]
  site_bodyid: npt.NDArray[np.int32]
  site_sameframe: npt.NDArray[np.uint8]
  site_size: npt.NDArray[np.float64]
  site_pos: jax.Array
  site_quat: jax.Array
  cam_mode: npt.NDArray[np.int32]
  cam_bodyid: npt.NDArray[np.int32]
  cam_targetbodyid: npt.NDArray[np.int32]
  cam_pos: jax.Array
  cam_quat: jax.Array
  cam_poscom0: jax.Array
  cam_pos0: jax.Array
  cam_mat0: jax.Array
  cam_fovy: jax.Array
  cam_resolution: npt.NDArray[np.int32]
  cam_sensorsize: npt.NDArray[np.float32]
  cam_intrinsic: jax.Array
  light_mode: npt.NDArray[np.int32]
  light_type: jax.Array
  light_active: jax.Array
  light_castshadow: jax.Array
  light_pos: jax.Array
  light_dir: jax.Array
  light_poscom0: jax.Array
  light_pos0: jax.Array
  light_dir0: jax.Array
  light_cutoff: jax.Array
  light_ambient: jax.Array
  light_attenuation: jax.Array
  light_diffuse: jax.Array
  light_exponent: jax.Array
  light_specular: jax.Array
  mesh_vertadr: npt.NDArray[np.int32]
  mesh_vertnum: npt.NDArray[np.int32]
  mesh_faceadr: npt.NDArray[np.int32]
  mesh_facenum: npt.NDArray[np.int32]
  mesh_bvhadr: npt.NDArray[np.int32]
  mesh_bvhnum: npt.NDArray[np.int32]
  mesh_octadr: npt.NDArray[np.int32]
  mesh_octnum: npt.NDArray[np.int32]
  mesh_normaladr: npt.NDArray[np.int32]
  mesh_normalnum: npt.NDArray[np.int32]
  mesh_graphadr: npt.NDArray[np.int32]
  mesh_vert: npt.NDArray[np.float32]
  mesh_normal: npt.NDArray[np.float32]
  mesh_face: npt.NDArray[np.int32]
  mesh_graph: npt.NDArray[np.int32]
  mesh_pos: npt.NDArray[np.float64]
  mesh_quat: npt.NDArray[np.float64]
  mesh_texcoordadr: npt.NDArray[np.int32]
  mesh_texcoordnum: npt.NDArray[np.int32]
  mesh_texcoord: npt.NDArray[np.float32]
  flex_vertadr: npt.NDArray[np.int32]
  flex_vertnum: npt.NDArray[np.int32]
  flex_interp: npt.NDArray[np.int32]
  flex_vert0: npt.NDArray[np.float64]
  flex_nodeadr: npt.NDArray[np.int32]
  flex_nodenum: npt.NDArray[np.int32]
  flex_nodebodyid: npt.NDArray[np.int32]
  flex_node0: npt.NDArray[np.float64]
  hfield_size: npt.NDArray[np.float64]
  hfield_nrow: npt.NDArray[np.int32]
  hfield_ncol: npt.NDArray[np.int32]
  hfield_adr: npt.NDArray[np.int32]
  hfield_data: jax.Array
  tex_type: npt.NDArray[np.int32]
  tex_height: npt.NDArray[np.int32]
  tex_width: npt.NDArray[np.int32]
  tex_nchannel: npt.NDArray[np.int32]
  tex_adr: npt.NDArray[np.int64]
  tex_data: npt.NDArray[np.uint8]
  mat_rgba: jax.Array
  mat_texid: jax.Array
  mat_emission: jax.Array
  mat_specular: jax.Array
  mat_shininess: jax.Array
  pair_dim: npt.NDArray[np.int32]
  pair_geom1: npt.NDArray[np.int32]
  pair_geom2: npt.NDArray[np.int32]
  pair_signature: npt.NDArray[np.int32]
  pair_solref: jax.Array
  pair_solreffriction: jax.Array
  pair_solimp: jax.Array
  pair_margin: jax.Array
  pair_gap: jax.Array
  pair_friction: jax.Array
  exclude_signature: npt.NDArray[np.int32]
  eq_type: npt.NDArray[np.int32]
  eq_obj1id: npt.NDArray[np.int32]
  eq_obj2id: npt.NDArray[np.int32]
  eq_objtype: npt.NDArray[np.int32]
  eq_active0: npt.NDArray[np.bool_]
  eq_solref: jax.Array
  eq_solimp: jax.Array
  eq_data: jax.Array
  tendon_type: npt.NDArray[np.int32]
  tendon_adr: npt.NDArray[np.int32]
  tendon_num: npt.NDArray[np.int32]
  tendon_limited: npt.NDArray[np.bool_]
  tendon_actfrclimited: npt.NDArray[np.bool_]
  tendon_solref_lim: jax.Array
  tendon_solimp_lim: jax.Array
  tendon_solref_fri: jax.Array
  tendon_solimp_fri: jax.Array
  tendon_range: jax.Array
  tendon_actfrcrange: jax.Array
  tendon_margin: jax.Array
  tendon_stiffness: jax.Array
  tendon_stiffnesspoly: jax.Array
  tendon_damping: jax.Array
  tendon_dampingpoly: jax.Array
  tendon_armature: jax.Array
  tendon_frictionloss: jax.Array
  tendon_lengthspring: jax.Array
  tendon_length0: jax.Array
  tendon_invweight0: jax.Array
  wrap_type: npt.NDArray[np.int32]
  wrap_objid: npt.NDArray[np.int32]
  wrap_prm: npt.NDArray[np.float64]
  actuator_trntype: npt.NDArray[np.int32]
  actuator_dyntype: npt.NDArray[np.int32]
  actuator_gaintype: npt.NDArray[np.int32]
  actuator_biastype: npt.NDArray[np.int32]
  actuator_trnid: npt.NDArray[np.int32]
  actuator_actadr: npt.NDArray[np.int32]
  actuator_actnum: npt.NDArray[np.int32]
  actuator_group: npt.NDArray[np.int32]
  actuator_ctrllimited: npt.NDArray[np.bool_]
  actuator_forcelimited: npt.NDArray[np.bool_]
  actuator_actlimited: npt.NDArray[np.bool_]
  actuator_dynprm: jax.Array
  actuator_gainprm: jax.Array
  actuator_biasprm: jax.Array
  actuator_actearly: npt.NDArray[np.bool_]
  actuator_ctrlrange: jax.Array
  actuator_forcerange: jax.Array
  actuator_actrange: jax.Array
  actuator_gear: jax.Array
  actuator_cranklength: jax.Array
  actuator_acc0: jax.Array
  actuator_lengthrange: jax.Array
  sensor_type: npt.NDArray[np.int32]
  sensor_datatype: npt.NDArray[np.int32]
  sensor_needstage: npt.NDArray[np.int32]
  sensor_objtype: npt.NDArray[np.int32]
  sensor_objid: npt.NDArray[np.int32]
  sensor_reftype: npt.NDArray[np.int32]
  sensor_refid: npt.NDArray[np.int32]
  sensor_intprm: npt.NDArray[np.int32]
  sensor_dim: npt.NDArray[np.int32]
  sensor_adr: npt.NDArray[np.int32]
  sensor_cutoff: npt.NDArray[np.float64]
  numeric_adr: npt.NDArray[np.int32]
  numeric_data: npt.NDArray[np.float64]
  tuple_adr: npt.NDArray[np.int32]
  tuple_size: npt.NDArray[np.int32]
  tuple_objtype: npt.NDArray[np.int32]
  tuple_objid: npt.NDArray[np.int32]
  tuple_objprm: npt.NDArray[np.float64]
  key_time: npt.NDArray[np.float64]
  key_qpos: npt.NDArray[np.float64]
  key_qvel: npt.NDArray[np.float64]
  key_act: npt.NDArray[np.float64]
  key_mpos: npt.NDArray[np.float64]
  key_mquat: npt.NDArray[np.float64]
  key_ctrl: npt.NDArray[np.float64]
  key_plastic: npt.NDArray[np.float64]
  name_bodyadr: npt.NDArray[np.int32]
  name_jntadr: npt.NDArray[np.int32]
  name_geomadr: npt.NDArray[np.int32]
  name_siteadr: npt.NDArray[np.int32]
  name_camadr: npt.NDArray[np.int32]
  name_flexadr: npt.NDArray[np.int32]
  name_meshadr: npt.NDArray[np.int32]
  name_hfieldadr: npt.NDArray[np.int32]
  name_pairadr: npt.NDArray[np.int32]
  name_eqadr: npt.NDArray[np.int32]
  name_tendonadr: npt.NDArray[np.int32]
  name_actuatoradr: npt.NDArray[np.int32]
  name_sensoradr: npt.NDArray[np.int32]
  name_numericadr: npt.NDArray[np.int32]
  name_tupleadr: npt.NDArray[np.int32]
  name_keyadr: npt.NDArray[np.int32]
  names: bytes
  signature: int
  _sizes: jax.Array
  _impl: ModelCPP | ModelJAX | mjxw_types.ModelWarp
  @property
  def impl(self) -> Impl: ...
  def __getattr__(self, name: str) -> Any: ...
  def bind(
    self, obj: mujoco.MjStruct | Iterable[mujoco.MjStruct]
  ) -> BindModel: ...

class Contact(PyTreeNode):
  """Result of collision detection functions."""

  dist: jax.Array
  pos: jax.Array
  frame: jax.Array
  includemargin: jax.Array
  friction: jax.Array
  solref: jax.Array
  solreffriction: jax.Array
  solimp: jax.Array
  dim: npt.NDArray[np.int32]
  geom1: jax.Array
  geom2: jax.Array
  geom: jax.Array
  efc_address: npt.NDArray[np.int64]

class DataJAX(PyTreeNode):
  """JAX-specific data."""

  ne: int
  nf: int
  nl: int
  nefc: int
  ncon: int
  solver_niter: jax.Array
  cinert: jax.Array
  ten_wrapadr: jax.Array
  ten_wrapnum: jax.Array
  ten_J: jax.Array
  wrap_obj: jax.Array
  wrap_xpos: jax.Array
  actuator_moment: jax.Array
  crb: jax.Array
  M: jax.Array
  qLD: jax.Array
  qLDiagInv: jax.Array
  ten_velocity: jax.Array
  actuator_velocity: jax.Array
  cacc: jax.Array
  cfrc_int: jax.Array
  cfrc_ext: jax.Array
  subtree_linvel: jax.Array
  subtree_angmom: jax.Array
  contact: Contact
  efc_type: jax.Array
  efc_J: jax.Array
  efc_pos: jax.Array
  efc_margin: jax.Array
  efc_frictionloss: jax.Array
  efc_D: jax.Array
  efc_aref: jax.Array
  efc_force: jax.Array

class Data(PyTreeNode):
  """Dynamic state that updates each step."""

  time: jax.Array
  qpos: jax.Array
  qvel: jax.Array
  act: jax.Array
  history: jax.Array
  qacc_warmstart: jax.Array
  plugin_state: jax.Array
  plastic: jax.Array
  ctrl: jax.Array
  qfrc_applied: jax.Array
  xfrc_applied: jax.Array
  eq_active: jax.Array
  mocap_pos: jax.Array
  mocap_quat: jax.Array
  qacc: jax.Array
  act_dot: jax.Array
  userdata: jax.Array
  sensordata: jax.Array
  xpos: jax.Array
  xquat: jax.Array
  xmat: jax.Array
  xipos: jax.Array
  ximat: jax.Array
  xanchor: jax.Array
  xaxis: jax.Array
  ten_length: jax.Array
  geom_xpos: jax.Array
  geom_xmat: jax.Array
  site_xpos: jax.Array
  site_xmat: jax.Array
  cam_xpos: jax.Array
  cam_xmat: jax.Array
  subtree_com: jax.Array
  cvel: jax.Array
  cdof: jax.Array
  cdof_dot: jax.Array
  qfrc_bias: jax.Array
  qfrc_gravcomp: jax.Array
  qfrc_fluid: jax.Array
  qfrc_passive: jax.Array
  qfrc_actuator: jax.Array
  actuator_force: jax.Array
  actuator_length: jax.Array
  qfrc_smooth: jax.Array
  qacc_smooth: jax.Array
  qfrc_constraint: jax.Array
  qfrc_inverse: jax.Array
  _impl: DataCPP | DataJAX | mjxw_types.DataWarp
  @property
  def impl(self) -> Impl: ...
  def __getattr__(self, name: str) -> Any: ...
  def bind(
    self,
    model: Model,
    obj: mujoco.MjStruct | Iterable[mujoco.MjStruct],
  ) -> BindData: ...
  def __getitem__(self, key: _Index) -> Data: ...
  def where(self, done: jax.Array, other: Data) -> Data: ...

def tree_path_to_attr_str(path: jax.tree_util.KeyPath) -> str: ...
