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

import dataclasses
import enum
from collections.abc import Callable, Iterator, Sequence
from typing import Generic, TypeVar, overload

import warp as wp

_Scalar = TypeVar('_Scalar', int, float)

MJ_MINVAL: float
MJ_MAXVAL: float
MJ_MINIMP: float
MJ_MAXIMP: float
MJ_MAXCONPAIR: int
MJ_MINMU: float
MJ_MINAWAKE: int
FLEX_STIFFNESS_3D: int
FLEX_INTERP_PHYS_JAC: bool
ROD_HINGE_SIZE: int
MJ_MAX_EPAHORIZON: int
MJ_MAX_EPAFACES: int
TILE_SIZE_JTDAJ_SPARSE: int
TILE_SIZE_JTDAJ_DENSE: int
M_BLOCK_DENSE_MAX: int
M_BLOCK_SCALAR_MAX: int
Q_LD_BLOCK_COMPACT: int
Q_LD_BLOCK_SPARSE: int
FLEX_INTERP_MAX_NODES: int

@dataclasses.dataclass
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

class BroadphaseType(enum.IntEnum):
  """Type of broadphase algorithm."""

  NXN = 0
  SAP_TILE = 1
  SAP_SEGMENTED = 2

class BroadphaseFilter(enum.IntFlag):
  """Bitmask specifying which collision functions to run during broadphase."""

  PLANE = 1
  SPHERE = 2
  AABB = 4
  OBB = 8

class OverflowType(enum.IntFlag):
  """Bitmask for physics and collision overflows."""

  NONE = 0
  NEFC = 1
  NJMAX_NNZ = 2
  BROADPHASE = 4
  NARROWPHASE = 8
  CCD = 16
  HFIELD = 32
  CONTACT_MATCH = 64
  NVMAX = 128
  EPA_HORIZON = 256
  ITERATIONS = 512
  LS_ITERATIONS = 1024
  TACTILE = 2048
  ALL = 4095

class CamLightType(enum.IntEnum):
  """Type of camera light."""

  FIXED = 0
  TRACK = 1
  TRACKCOM = 2
  TARGETBODY = 3
  TARGETBODYCOM = 4

class ProjectionType(enum.IntEnum):
  """Type of camera projection."""

  PERSPECTIVE = 0
  ORTHOGRAPHIC = 1

class Stage(enum.IntEnum):
  """Computation stage."""

  POS = 1
  VEL = 2
  ACC = 3

class DataType(enum.IntFlag):
  """Sensor data types."""

  REAL = 0
  POSITIVE = 1

class DeterminismType(enum.IntFlag):
  """Bitmask for deterministic execution."""

  NONE = 0
  CONTACTS = 1
  CONSTRAINT = 2
  ATOMICS = 4
  ALL = 7

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
  FILTERPARENT = 1024
  ACTUATION = 2048
  REFSAFE = 4096
  SENSOR = 8192
  EULERDAMP = 32768
  NATIVECCD = 131072
  ISLAND = 262144
  MULTICCD = 524288

class EnableBit(enum.IntFlag):
  """Enable optional feature bitflags."""

  ENERGY = 2
  INVDISCRETE = 8
  SLEEP = 16

class SleepPolicy(enum.IntEnum):
  """Per-tree sleep policy."""

  AUTO = 0
  AUTO_NEVER = 1
  AUTO_ALLOWED = 2

class SleepState(enum.IntEnum):
  """Sleep state for bodies."""

  STATIC = -1
  ASLEEP = 0
  AWAKE = 1

class TrnType(enum.IntEnum):
  """Type of actuator transmission."""

  JOINT = 0
  JOINTINPARENT = 1
  SLIDERCRANK = 2
  TENDON = 3
  BODY = 5
  SITE = 4

class DynType(enum.IntEnum):
  """Type of actuator dynamics."""

  NONE = 0
  INTEGRATOR = 1
  FILTER = 2
  FILTEREXACT = 3
  MUSCLE = 4
  USER = 7
  DCMOTOR = 5

class GainType(enum.IntEnum):
  """Type of actuator gain."""

  FIXED = 0
  AFFINE = 1
  MUSCLE = 2
  USER = 6
  DCMOTOR = 3
  SO3 = 4

class BiasType(enum.IntEnum):
  """Type of actuator bias."""

  NONE = 0
  AFFINE = 1
  MUSCLE = 2
  USER = 5
  DCMOTOR = 3

class CtrlInput(enum.IntFlag):
  """Control input signature bitflags for actuators."""

  POS = 1
  VEL = 2
  FF = 4
  VOLTAGE = 8
  NONE = 16

class CtrlChart(enum.IntEnum):
  """Orientation chart type for SO(3) actuators."""

  EXPMAP = 1
  QUAT = 2

class JointType(enum.IntEnum):
  """Type of degree of freedom."""

  FREE = 0
  BALL = 1
  SLIDE = 2
  HINGE = 3

class ConeType(enum.IntEnum):
  """Type of friction cone."""

  PYRAMIDAL = 0
  ELLIPTIC = 1

class IntegratorType(enum.IntEnum):
  """Integrator mode."""

  EULER = 0
  RK4 = 1
  IMPLICITFAST = 3
  IMPLICIT = 2
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
  SDF = 8
  FLEX = 105
  TRIANGLE = 999

class CollisionType(enum.IntEnum):
  """Type of narrowphase collision."""

  PRIMITIVE = 0
  CONVEX = 1
  SDF = 2

class SolverType(enum.IntEnum):
  """Constraint solver algorithm."""

  CG = 1
  NEWTON = 2

class ConstraintState(enum.IntEnum):
  """State of constraint."""

  SATISFIED = 0
  QUADRATIC = 1
  LINEARNEG = 2
  LINEARPOS = 3
  CONE = 4

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

class SensorType(enum.IntEnum):
  """Type of sensor."""

  MAGNETOMETER = 6
  CAMPROJECTION = 8
  RANGEFINDER = 7
  JOINTPOS = 9
  TENDONPOS = 11
  ACTUATORPOS = 13
  BALLQUAT = 18
  JOINTLIMITPOS = 20
  TENDONLIMITPOS = 23
  FRAMEPOS = 26
  FRAMEXAXIS = 28
  FRAMEYAXIS = 29
  FRAMEZAXIS = 30
  FRAMEQUAT = 27
  SUBTREECOM = 35
  GEOMDIST = 39
  GEOMNORMAL = 40
  GEOMFROMTO = 41
  INSIDESITE = 38
  E_POTENTIAL = 43
  E_KINETIC = 44
  CLOCK = 45
  VELOCIMETER = 2
  GYRO = 3
  JOINTVEL = 10
  TENDONVEL = 12
  ACTUATORVEL = 14
  BALLANGVEL = 19
  JOINTLIMITVEL = 21
  TENDONLIMITVEL = 24
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
  TENDONACTFRC = 17
  JOINTACTFRC = 16
  JOINTLIMITFRC = 22
  TENDONLIMITFRC = 25
  FRAMELINACC = 33
  FRAMEANGACC = 34
  TACTILE = 46
  USER = 48

class RayDataField(enum.IntEnum):
  """Ray data fields for rangefinder sensors."""

  DIST = 0
  DIR = 1
  ORIGIN = 2
  POINT = 3
  NORMAL = 4
  DEPTH = 5

class ConDataField(enum.IntEnum):
  """Contact data fields for contact sensors."""

  FOUND = 0
  FORCE = 1
  TORQUE = 2
  DIST = 3
  POS = 4
  NORMAL = 5
  TANGENT = 6
  LINVEL = 7
  ANGVEL = 8

class ObjType(enum.IntEnum):
  """Type of object."""

  UNKNOWN = 0
  BODY = 1
  XBODY = 2
  JOINT = 3
  GEOM = 5
  FLEX = 9
  SITE = 6
  CAMERA = 7
  TENDON = 18

class EqType(enum.IntEnum):
  """Type of equality constraint."""

  CONNECT = 0
  WELD = 1
  JOINT = 2
  TENDON = 3
  FLEX = 4
  FLEXSTRAIN = 6

class TendonType(enum.IntEnum):
  """Type of tendon."""

  FIXED = 0
  SPATIAL = 1

class WrapType(enum.IntEnum):
  """Type of tendon wrapping object."""

  JOINT = 1
  PULLEY = 2
  SITE = 3
  SPHERE = 4
  CYLINDER = 5

class State(enum.IntEnum):
  """State component elements as integer bitflags."""

  TIME = 1
  QPOS = 2
  QVEL = 4
  ACT = 8
  HISTORY = 16
  WARMSTART = 32
  CTRL = 64
  QFRC_APPLIED = 128
  XFRC_APPLIED = 256
  EQ_ACTIVE = 512
  MOCAP_POS = 1024
  MOCAP_QUAT = 2048
  USERDATA = 4096
  PLASTIC = 16384
  NSTATE = 15
  PHYSICS = 16414
  FULLPHYSICS = 24607
  USER = 8128
  INTEGRATION = 32767

class SetConstGroup(enum.IntFlag):
  """Groups of set_const computations."""

  NONE = 0
  FIXED = 1
  KIN0 = 2
  INERT0 = 4
  ALL = 7

class _Vec(Generic[_Scalar]):
  def __init__(self, *args: _Scalar | Sequence[_Scalar]) -> None: ...
  def __len__(self) -> int: ...
  def __iter__(self) -> Iterator[_Scalar]: ...
  @overload
  def __getitem__(self, key: int) -> _Scalar: ...
  @overload
  def __getitem__(self, key: slice) -> _Vec[_Scalar]: ...
  def __setitem__(self, key: int, value: _Scalar) -> None: ...
  def __mul__(self, scalar: _Scalar) -> _Vec[_Scalar]: ...
  def __rmul__(self, scalar: _Scalar) -> _Vec[_Scalar]: ...
  def __truediv__(self, scalar: _Scalar) -> _Vec[_Scalar]: ...
  def __neg__(self) -> _Vec[_Scalar]: ...
  def __eq__(self, other: object) -> bool: ...

class _Mat(Generic[_Scalar]):
  def __init__(
    self, *args: _Scalar | Sequence[_Scalar] | _Vec[_Scalar]
  ) -> None: ...
  def __len__(self) -> int: ...
  @overload
  def __getitem__(self, key: int) -> _Vec[_Scalar]: ...
  @overload
  def __getitem__(self, key: tuple[int, int]) -> _Scalar: ...
  @overload
  def __setitem__(self, key: int, value: _Vec[_Scalar]) -> None: ...
  @overload
  def __setitem__(self, key: tuple[int, int], value: _Scalar) -> None: ...
  def get_row(self, i: int) -> _Vec[_Scalar]: ...
  def get_col(self, i: int) -> _Vec[_Scalar]: ...
  def set_row(self, i: int, row: _Vec[_Scalar]) -> None: ...
  def set_col(self, i: int, col: _Vec[_Scalar]) -> None: ...
  def __eq__(self, other: object) -> bool: ...

class vec5f(_Vec[float]): ...
class vec6f(_Vec[float]): ...
class vec6i(_Vec[int]): ...
class vec8f(_Vec[float]): ...
class vec8i(_Vec[int]): ...
class vec16f(_Vec[float]): ...
class vec16i(_Vec[int]): ...
class vec10f(_Vec[float]): ...
class vec11f(_Vec[float]): ...
class vec12f(_Vec[float]): ...
class vec_pluginattr(_Vec[float]): ...
class mat23f(_Mat[float]): ...
class mat43f(_Mat[float]): ...
class mat53f(_Mat[float]): ...
class mat63f(_Mat[float]): ...
class mat55f(_Mat[float]): ...
class mat66f(_Mat[float]): ...
class mat113f(_Mat[float]): ...

vec5 = vec5f
vec6 = vec6f
vec8 = vec8f
vec10 = vec10f
vec11 = vec11f
vec12 = vec12f
vec16 = vec16f
vec128 = vec_pluginattr
mat23 = mat23f
mat43 = mat43f
mat53 = mat53f
mat63 = mat63f
mat55 = mat55f
mat66 = mat66f
mat113 = mat113f

def array(*args: str | int | type) -> wp.array: ...

@dataclasses.dataclass
class Option:
  """Physics options."""

  timestep: wp.array[float]
  tolerance: wp.array[float]
  ls_tolerance: wp.array[float]
  noslip_tolerance: wp.array[float]
  ccd_tolerance: wp.array[float]
  sleep_tolerance: wp.array[float]
  gravity: wp.array[wp.vec3]
  wind: wp.array[wp.vec3]
  magnetic: wp.array[wp.vec3]
  density: wp.array[float]
  viscosity: wp.array[float]
  integrator: int
  cone: int
  solver: int
  iterations: int
  ls_iterations: int
  noslip_iterations: int
  ccd_iterations: int
  disableflags: int
  enableflags: int
  sdf_initpoints: int
  sdf_iterations: int
  impratio_invsqrt: wp.array[float]
  broadphase: BroadphaseType
  broadphase_filter: BroadphaseFilter
  graph_conditional: bool
  run_collision_detection: bool
  run_rne_postconstraint: bool
  run_qacc_smooth: bool
  contact_sensor_maxmatch: int
  warn_overflow: int = ...
  deterministic: int = ...
  @property
  def ls_parallel(self) -> bool: ...
  @ls_parallel.setter
  def ls_parallel(self, value: bool) -> None: ...
  @property
  def ls_parallel_min_step(self) -> float: ...
  @ls_parallel_min_step.setter
  def ls_parallel_min_step(self, value: float) -> None: ...

@dataclasses.dataclass
class TileSet:
  """Tiling configuration for decomposable block diagonal matrix."""

  adr: wp.array[int]
  size: int
  elemid: wp.array[int] = ...
  def __eq__(self, other: object) -> bool: ...
  def __hash__(self) -> int: ...

@dataclasses.dataclass
class Callback:
  """Callbacks for custom physics behavior."""

  passive: Callable[[Model, Data], None] | None = ...
  control: Callable[[Model, Data], None] | None = ...
  act_dyn: Callable[[Model, Data], None] | None = ...
  act_gain: Callable[[Model, Data], None] | None = ...
  act_bias: Callable[[Model, Data], None] | None = ...
  sensor: Callable[[Model, Data, Stage], None] | None = ...
  contactfilter: Callable[[Model, Data], None] | None = ...

@dataclasses.dataclass
class Model:
  """Model definition and parameters."""

  nq: int
  nv: int
  nu: int
  nactuator: int
  na: int
  nbody: int
  noct: int
  njnt: int
  ntree: int
  nM: int
  nC: int
  nD: int
  ngeom: int
  nsite: int
  ncam: int
  nlight: int
  nflex: int
  nflexnode: int
  nflexvert: int
  nflexedge: int
  nflexelem: int
  nflexelemdata: int
  nflexstiffness: int
  nflexbending: int
  nefm0dof: int
  nefm0L: int
  nflexelemedge: int
  nflexshelldata: int
  nJfe: int
  nmesh: int
  nmeshvert: int
  nmeshnormal: int
  nmeshface: int
  nmeshgraph: int
  nmeshpoly: int
  nmeshpolyvert: int
  nmeshpolymap: int
  nhfield: int
  nhfielddata: int
  nmat: int
  npair: int
  nexclude: int
  neq: int
  ntendon: int
  nJten: int
  nwrap: int
  nsensor: int
  nkey: int
  nmocap: int
  nplugin: int
  nplastic: int
  nJmom: int
  npolygonmax: int
  nmeshdegmax: int
  nuserdata: int
  nsensordata: int
  nhistory: int
  opt: Option
  qpos0: wp.array[float]
  qpos_spring: wp.array[float]
  body_parentid: wp.array[int]
  body_rootid: wp.array[int]
  body_weldid: wp.array[int]
  body_mocapid: wp.array[int]
  body_jntnum: wp.array[int]
  body_jntadr: wp.array[int]
  body_dofnum: wp.array[int]
  body_dofadr: wp.array[int]
  body_treeid: wp.array[int]
  body_geomnum: wp.array[int]
  body_geomadr: wp.array[int]
  body_simple: wp.array[int]
  body_pos: wp.array[wp.vec3]
  body_quat: wp.array[wp.quat]
  body_ipos: wp.array[wp.vec3]
  body_iquat: wp.array[wp.quat]
  body_mass: wp.array[float]
  body_subtreemass: wp.array[float]
  body_inertia: wp.array[wp.vec3]
  body_invweight0: wp.array[wp.vec2]
  body_gravcomp: wp.array[float]
  body_contype: wp.array[int]
  body_conaffinity: wp.array[int]
  oct_child: wp.array[vec8i]
  oct_aabb: wp.array[wp.vec3]
  oct_coeff: wp.array[vec8]
  jnt_type: wp.array[int]
  jnt_qposadr: wp.array[int]
  jnt_dofadr: wp.array[int]
  jnt_bodyid: wp.array[int]
  jnt_actuatorid: wp.array[int]
  jnt_plasticadr: wp.array[int]
  jnt_limited: wp.array[int]
  jnt_actfrclimited: wp.array[bool]
  jnt_actgravcomp: wp.array[int]
  jnt_solref: wp.array[wp.vec2]
  jnt_solimp: wp.array[vec5]
  jnt_pos: wp.array[wp.vec3]
  jnt_axis: wp.array[wp.vec3]
  jnt_stiffness: wp.array[float]
  jnt_stiffnesspoly: wp.array[wp.vec2]
  jnt_springyield: wp.array[float]
  jnt_range: wp.array[wp.vec2]
  jnt_actfrcrange: wp.array[wp.vec2]
  jnt_margin: wp.array[float]
  dof_bodyid: wp.array[int]
  dof_jntid: wp.array[int]
  dof_parentid: wp.array[int]
  dof_treeid: wp.array[int]
  dof_Madr: wp.array[int]
  dof_solref: wp.array[wp.vec2]
  dof_solimp: wp.array[vec5]
  dof_frictionloss: wp.array[float]
  dof_armature: wp.array[float]
  dof_damping: wp.array[float]
  dof_dampingpoly: wp.array[wp.vec2]
  dof_invweight0: wp.array[float]
  dof_length: wp.array[float]
  tree_bodynum: wp.array[int]
  tree_dofadr: wp.array[int]
  tree_dofnum: wp.array[int]
  tree_sleep_policy: wp.array[int]
  geom_type: wp.array[int]
  geom_contype: wp.array[int]
  geom_conaffinity: wp.array[int]
  geom_condim: wp.array[int]
  geom_bodyid: wp.array[int]
  geom_dataid: wp.array[int]
  geom_matid: wp.array[int]
  geom_group: wp.array[int]
  geom_priority: wp.array[int]
  geom_solmix: wp.array[float]
  geom_solref: wp.array[wp.vec2]
  geom_solimp: wp.array[vec5]
  geom_size: wp.array[wp.vec3]
  geom_aabb: wp.array[wp.vec3]
  geom_rbound: wp.array[float]
  geom_pos: wp.array[wp.vec3]
  geom_quat: wp.array[wp.quat]
  geom_friction: wp.array[wp.vec3]
  geom_margin: wp.array[float]
  geom_gap: wp.array[float]
  geom_surfacevel: wp.array[vec6]
  geom_adhesion: wp.array[float]
  geom_fluid: wp.array[float]
  geom_rgba: wp.array[wp.vec4]
  site_type: wp.array[int]
  site_bodyid: wp.array[int]
  site_size: wp.array[wp.vec3]
  site_pos: wp.array[wp.vec3]
  site_quat: wp.array[wp.quat]
  cam_mode: wp.array[int]
  cam_bodyid: wp.array[int]
  cam_targetbodyid: wp.array[int]
  cam_pos: wp.array[wp.vec3]
  cam_quat: wp.array[wp.quat]
  cam_poscom0: wp.array[wp.vec3]
  cam_pos0: wp.array[wp.vec3]
  cam_mat0: wp.array[wp.mat33]
  cam_projection: wp.array[int]
  cam_fovy: wp.array[float]
  cam_resolution: wp.array[wp.vec2i]
  cam_sensorsize: wp.array[wp.vec2]
  cam_intrinsic: wp.array[wp.vec4]
  light_mode: wp.array[int]
  light_bodyid: wp.array[int]
  light_targetbodyid: wp.array[int]
  light_type: wp.array[int]
  light_castshadow: wp.array[bool]
  light_active: wp.array[bool]
  light_pos: wp.array[wp.vec3]
  light_dir: wp.array[wp.vec3]
  light_poscom0: wp.array[wp.vec3]
  light_pos0: wp.array[wp.vec3]
  light_dir0: wp.array[wp.vec3]
  light_attenuation: wp.array[wp.vec3]
  light_cutoff: wp.array[float]
  light_exponent: wp.array[float]
  light_ambient: wp.array[wp.vec3]
  light_diffuse: wp.array[wp.vec3]
  light_specular: wp.array[wp.vec3]
  flex_contype: wp.array[int]
  flex_conaffinity: wp.array[int]
  flex_condim: wp.array[int]
  flex_priority: wp.array[int]
  flex_solmix: wp.array[float]
  flex_solref: wp.array[wp.vec2]
  flex_solimp: wp.array[vec5]
  flex_friction: wp.array[wp.vec3]
  flex_margin: wp.array[float]
  flex_gap: wp.array[float]
  flex_selfcollide: wp.array[int]
  flex_activelayers: wp.array[int]
  flex_passive: wp.array[int]
  flex_dim: wp.array[int]
  flex_interp: wp.array[int]
  flex_cellnum: wp.array[wp.vec3i]
  flex_nodeadr: wp.array[int]
  flex_nodenum: wp.array[int]
  flex_vertadr: wp.array[int]
  flex_vertnum: wp.array[int]
  flex_edgeadr: wp.array[int]
  flex_edgenum: wp.array[int]
  flex_elemadr: wp.array[int]
  flex_elemnum: wp.array[int]
  flex_elemdataadr: wp.array[int]
  flex_stiffnessadr: wp.array[int]
  flex_elemedgeadr: wp.array[int]
  flex_bendingadr: wp.array[int]
  flex_plasticadr: wp.array[int]
  flex_plasticmode: wp.array[int]
  flex_shellnum: wp.array[int]
  flex_shelldataadr: wp.array[int]
  flex_nodebodyid: wp.array[int]
  flex_vertbodyid: wp.array[int]
  flex_vertcell: wp.array[int]
  flex_edge: wp.array[wp.vec2i]
  flex_edgeflap: wp.array[wp.vec2i]
  flex_elem: wp.array[int]
  flex_elemedge: wp.array[int]
  flex_elemlayer: wp.array[int]
  flex_shell: wp.array[int]
  flex_vert: wp.array[wp.vec3]
  flex_vert0: wp.array[wp.vec3]
  flex_node: wp.array[wp.vec3]
  flex_node0: wp.array[wp.vec3]
  flexedge_length0: wp.array[float]
  flexedge_invweight0: wp.array[float]
  flex_radius: wp.array[float]
  flex_size: wp.array[wp.vec3]
  flex_stiffness: wp.array[float]
  flex_bending: wp.array[float]
  efm0_dofid: wp.array[int]
  efm0_L_rownnz: wp.array[int]
  efm0_L_rowadr: wp.array[int]
  efm0_L_colind: wp.array[int]
  efm0_L: wp.array[float]
  flex_damping: wp.array[float]
  flex_yield: wp.array[wp.vec2]
  flex_edgestiffness: wp.array[float]
  flex_edgedamping: wp.array[float]
  flex_edgeequality: wp.array[int]
  flex_rigid: wp.array[bool]
  flexedge_rigid: wp.array[bool]
  flex_centered: wp.array[bool]
  flexedge_J_rownnz: wp.array[int]
  flexedge_J_rowadr: wp.array[int]
  flexedge_J_colind: wp.array[int]
  mesh_vertadr: wp.array[int]
  mesh_vertnum: wp.array[int]
  mesh_faceadr: wp.array[int]
  mesh_octadr: wp.array[int]
  mesh_normaladr: wp.array[int]
  mesh_normalnum: wp.array[int]
  mesh_graphadr: wp.array[int]
  mesh_vert: wp.array[wp.vec3]
  mesh_normal: wp.array[wp.vec3]
  mesh_face: wp.array[wp.vec3i]
  mesh_graph: wp.array[int]
  mesh_pos: wp.array[wp.vec3]
  mesh_quat: wp.array[wp.quat]
  mesh_polynum: wp.array[int]
  mesh_polyadr: wp.array[int]
  mesh_polynormal: wp.array[wp.vec3]
  mesh_polyvertadr: wp.array[int]
  mesh_polyvertnum: wp.array[int]
  mesh_polyvert: wp.array[int]
  mesh_polymapadr: wp.array[int]
  mesh_polymapnum: wp.array[int]
  mesh_polymap: wp.array[int]
  hfield_size: wp.array[wp.vec4]
  hfield_nrow: wp.array[int]
  hfield_ncol: wp.array[int]
  hfield_adr: wp.array[int]
  hfield_data: wp.array[float]
  mat_texid: wp.array[int]
  mat_texuniform: wp.array[bool]
  mat_texrepeat: wp.array[wp.vec2]
  mat_emission: wp.array[float]
  mat_specular: wp.array[float]
  mat_shininess: wp.array[float]
  mat_rgba: wp.array[wp.vec4]
  pair_dim: wp.array[int]
  pair_geom1: wp.array[int]
  pair_geom2: wp.array[int]
  pair_solref: wp.array[wp.vec2]
  pair_solreffriction: wp.array[wp.vec2]
  pair_solimp: wp.array[vec5]
  pair_margin: wp.array[float]
  pair_gap: wp.array[float]
  pair_adhesion: wp.array[float]
  pair_friction: wp.array[vec5]
  exclude_signature: wp.array[int]
  eq_type: wp.array[int]
  eq_obj1id: wp.array[int]
  eq_obj2id: wp.array[int]
  eq_objtype: wp.array[int]
  eq_active0: wp.array[bool]
  eq_solref: wp.array[wp.vec2]
  eq_solimp: wp.array[vec5]
  eq_data: wp.array[vec12]
  tendon_type: wp.array[int]
  tendon_adr: wp.array[int]
  tendon_num: wp.array[int]
  tendon_actuatorid: wp.array[int]
  tendon_plasticadr: wp.array[int]
  ten_J_rownnz: wp.array[int]
  ten_J_rowadr: wp.array[int]
  ten_J_colind: wp.array[int]
  tendon_limited: wp.array[int]
  tendon_actfrclimited: wp.array[bool]
  tendon_solref_lim: wp.array[wp.vec2]
  tendon_solimp_lim: wp.array[vec5]
  tendon_solref_fri: wp.array[wp.vec2]
  tendon_solimp_fri: wp.array[vec5]
  tendon_range: wp.array[wp.vec2]
  tendon_actfrcrange: wp.array[wp.vec2]
  tendon_margin: wp.array[float]
  tendon_stiffness: wp.array[float]
  tendon_stiffnesspoly: wp.array[wp.vec2]
  tendon_damping: wp.array[float]
  tendon_dampingpoly: wp.array[wp.vec2]
  tendon_armature: wp.array[float]
  tendon_frictionloss: wp.array[float]
  tendon_lengthspring: wp.array[wp.vec2]
  tendon_springyield: wp.array[float]
  tendon_length0: wp.array[float]
  tendon_invweight0: wp.array[float]
  wrap_type: wp.array[int]
  wrap_objid: wp.array[int]
  wrap_prm: wp.array[float]
  actuator_trntype: wp.array[int]
  actuator_dyntype: wp.array[int]
  actuator_gaintype: wp.array[int]
  actuator_biastype: wp.array[int]
  actuator_ctrladr: wp.array[int]
  actuator_ctrlnum: wp.array[int]
  actuator_ctrlspec: wp.array[int]
  actuator_actadr: wp.array[int]
  actuator_actnum: wp.array[int]
  actuator_trnid: wp.array[wp.vec2i]
  actuator_cranklength: wp.array[float]
  actuator_dynprm: wp.array[vec10]
  actuator_gainprm: wp.array[vec10]
  actuator_biasprm: wp.array[vec10]
  actuator_actlimited: wp.array[bool]
  actuator_actrange: wp.array[wp.vec2]
  actuator_actearly: wp.array[bool]
  actuator_history: wp.array[wp.vec2i]
  actuator_historyadr: wp.array[int]
  actuator_delay: wp.array[float]
  actuator_damping: wp.array[float]
  actuator_dampingpoly: wp.array[wp.vec2]
  actuator_armature: wp.array[float]
  actuator_forcelimited: wp.array[bool]
  actuator_forcerange: wp.array[wp.vec2]
  actuator_ctrllimited: wp.array[bool]
  actuator_ctrlrange: wp.array[wp.vec2]
  actuator_gear: wp.array[wp.spatial_vectorf]
  actuator_acc0: wp.array[float]
  actuator_lengthrange: wp.array[wp.vec2]
  sensor_type: wp.array[int]
  sensor_datatype: wp.array[int]
  sensor_objtype: wp.array[int]
  sensor_objid: wp.array[int]
  sensor_reftype: wp.array[int]
  sensor_refid: wp.array[int]
  sensor_intprm: wp.array[int]
  sensor_dim: wp.array[int]
  sensor_adr: wp.array[int]
  sensor_cutoff: wp.array[float]
  sensor_history: wp.array[wp.vec2i]
  sensor_historyadr: wp.array[int]
  sensor_delay: wp.array[float]
  sensor_interval: wp.array[wp.vec2]
  plugin: wp.array[int]
  plugin_attr: wp.array[vec_pluginattr]
  key_time: wp.array[float]
  key_qpos: wp.array[float]
  key_qvel: wp.array[float]
  key_act: wp.array[float]
  key_mpos: wp.array[wp.vec3]
  key_mquat: wp.array[wp.quat]
  key_ctrl: wp.array[float]
  key_plastic: wp.array[float]
  M_rownnz: wp.array[int]
  M_rowadr: wp.array[int]
  M_colind: wp.array[int]
  mapM2M: wp.array[int]
  D_rownnz: wp.array[int]
  D_rowadr: wp.array[int]
  D_diag: wp.array[int]
  D_colind: wp.array[int]
  mapM2D: wp.array[int]
  mapD2M: wp.array[int]
  nefmK: int
  nefmdof: int
  efm_K_rownnz: wp.array[int]
  efm_K_rowadr: wp.array[int]
  efm_K_colind: wp.array[int]
  efm_dofid: wp.array[int]
  callback: Callback
  nbranch: int
  nv_pad: int
  nacttrnbody: int
  nsensorcollision: int
  nsensortaxel: int
  ntactileweld: int
  nrangefinder: int
  nmaxcondim: int
  nmaxpyramid: int
  nflexintcell: int
  is_sparse: bool
  qLD_block_total: int
  qLD_block_adr: wp.array[int]
  flg_adhesion: bool
  has_fluid: bool
  flg_surfacevel: bool
  has_sdf_geom: bool
  has_flex_selfcollide: bool
  has_flex_passive: bool
  has_flex_snh: bool
  has_non_simple_flex: bool
  has_flex_edge_stiff_damp: bool
  has_tendon_stiffness: bool
  has_tendon_damping: bool
  has_efm_actuator: bool
  has_actuator_wrap: bool
  has_actuator_dcmotor: bool
  efm0_active: bool
  flex_interp_assemblable: bool
  has_unsupported_flex_interp: bool
  has_ellipsoid_geom: bool
  has_plane_geom: bool
  has_1d_flex: bool
  has_2d_flex: bool
  has_3d_flex: bool
  max_flex_dim: int
  block_dim: BlockDim
  body_tree: tuple[wp.array[int], ...]
  body_subtreenum: wp.array[int]
  body_branches: wp.array[int]
  body_branch_start: wp.array[int]
  mocap_bodyid: wp.array[int]
  body_fluid_ellipsoid: wp.array[bool]
  body_is_free: wp.array[bool]
  body_fluid_ellipsoid_adr: wp.array[int]
  body_fluid_box_adr: wp.array[int]
  body_freeadr: wp.array[int]
  jnt_limited_slide_hinge_adr: wp.array[int]
  jnt_limited_ball_adr: wp.array[int]
  body_isdofancestor: wp.array[int]
  dof_tri_row: wp.array[int]
  dof_tri_col: wp.array[int]
  nxn_geom_pair: wp.array[wp.vec2i]
  nxn_geom_pair_filtered: wp.array[wp.vec2i]
  nxn_pairid: wp.array[wp.vec2i]
  nxn_pairid_filtered: wp.array[wp.vec2i]
  geom_pair_type_count: tuple[int, ...]
  geom_plugin_index: wp.array[int]
  eq_connect_adr: wp.array[int]
  eq_wld_adr: wp.array[int]
  eq_jnt_adr: wp.array[int]
  eq_ten_adr: wp.array[int]
  eq_flex_adr: wp.array[int]
  eq_flexstrain_adr: wp.array[int]
  tendon_jnt_adr: wp.array[int]
  tendon_site_pair_adr: wp.array[int]
  tendon_geom_adr: wp.array[int]
  tendon_limited_adr: wp.array[int]
  max_ten_J_rownnz: int
  ten_wrapadr_site: wp.array[int]
  ten_wrapnum_site: wp.array[int]
  wrap_jnt_adr: wp.array[int]
  wrap_site_adr: wp.array[int]
  wrap_site_pair_adr: wp.array[int]
  wrap_geom_adr: wp.array[int]
  wrap_pulley_scale: wp.array[float]
  actuator_trntype_body_adr: wp.array[int]
  sensor_pos_adr: wp.array[int]
  sensor_limitpos_adr: wp.array[int]
  sensor_vel_adr: wp.array[int]
  sensor_limitvel_adr: wp.array[int]
  sensor_acc_adr: wp.array[int]
  sensor_stage_pos_adr: wp.array[int]
  sensor_stage_vel_adr: wp.array[int]
  sensor_stage_acc_adr: wp.array[int]
  sensor_rangefinder_adr: wp.array[int]
  rangefinder_sensor_adr: wp.array[int]
  sensor_collision_start_adr: wp.array[int]
  collision_sensor_adr: wp.array[int]
  sensor_touch_adr: wp.array[int]
  sensor_limitfrc_adr: wp.array[int]
  sensor_e_potential: bool
  sensor_e_kinetic: bool
  sensor_tendonactfrc_adr: wp.array[int]
  sensor_subtree_vel: bool
  sensor_contact_pos_adr: wp.array[int]
  sensor_contact_vel_adr: wp.array[int]
  sensor_contact_acc_adr: wp.array[int]
  sensor_rne_postconstraint: bool
  sensor_rangefinder_bodyid: wp.array[int]
  weld_tactile_id: wp.array[int]
  taxel_vertadr: wp.array[int]
  taxel_sensorid: wp.array[int]
  M_tiles: tuple[TileSet, ...]
  qLD_updates: tuple[wp.array[wp.vec3i], ...]
  qLD_all_updates: wp.array[wp.vec3i]
  qLD_level_offsets: wp.array[int]
  M_fullm_i: wp.array[int]
  M_fullm_j: wp.array[int]
  M_elemid: wp.array[int]
  M_hinit_i: wp.array[int]
  qD_fullm_i: wp.array[int]
  qD_fullm_j: wp.array[int]
  M_mulm_rowadr: wp.array[int]
  M_mulm_col: wp.array[int]
  M_mulm_madr: wp.array[int]
  flex_elemflexid: wp.array[int]
  flex_edgeflexid: wp.array[int]
  flex_shellflexid: wp.array[int]
  flex_vertflexid: wp.array[int]
  flex_shelladr: wp.array[int]
  flex_faceadr: wp.array[int]
  flex_simple: wp.array[bool]
  flex_efcnum: wp.array[int]
  flex_efcnnz: wp.array[int]
  flexedge_efcadr: wp.array[int]
  flexedge_efcnnzadr: wp.array[int]
  flex_cell_map: wp.array[wp.vec4i]
  flexstrain_J_rownnz: wp.array[int]
  flexstrain_J_rowadr: wp.array[int]
  flexstrain_J_colind: wp.array[int]
  neq_flexstrain: int
  nJfs: int
  nflexbend_interp: int
  flex_bend_interp_map: wp.array[wp.vec2i]
  nflexface: int
  flex_face_map: wp.array[wp.vec2i]
  flex_face: wp.array[int]
  nflexrodhinge: int
  flex_rodhinge_map: wp.array[wp.vec2i]
  nflexrodedge: int
  efm_con_max_dof: int
  flex_rod_is_chain: wp.array[bool]
  flex_rod_has_twist: wp.array[bool]
  flex_rod_is_box: wp.array[bool]
  flex_rod_root_clamped: wp.array[bool]
  flex_rod_tail_clamped: wp.array[bool]
  flex_eff_possible: wp.array[bool]
  flex_rod_box_size: wp.array[wp.vec2]
  flex_rod_hingeadr: wp.array[int]
  flex_rod_edgeadr: wp.array[int]
  flex_rod_edgenum: wp.array[int]
  flex_rodedge_vert: wp.array[wp.vec2i]
  flex_rodedge_jnt: wp.array[int]
  flex_elem_rodedge: wp.array[wp.vec2i]
  flex_vert_rodelem: wp.array[int]

class ContactType(enum.IntFlag):
  """Type of contact."""

  CONSTRAINT = 1
  SENSOR = 2
  PASSIVE = 4

@dataclasses.dataclass
class Contact:
  """Contact data."""

  dist: wp.array[float]
  pos: wp.array[wp.vec3]
  frame: wp.array[wp.mat33]
  includemargin: wp.array[float]
  friction: wp.array[vec5]
  solref: wp.array[wp.vec2]
  solreffriction: wp.array[wp.vec2]
  solimp: wp.array[vec5]
  dim: wp.array[int]
  geom: wp.array[wp.vec2i]
  flex: wp.array[wp.vec2i]
  elem: wp.array[wp.vec2i]
  vert: wp.array[wp.vec2i]
  efc_address: wp.array[int]
  worldid: wp.array[int]
  type: wp.array[int]
  geomcollisionid: wp.array[int]
  adhesion: wp.array[float]
  surfacevel: wp.array[wp.vec3]

@dataclasses.dataclass
class Constraint:
  """Constraint data."""

  type: wp.array[int]
  id: wp.array[int]
  jtdaj_adr: wp.array[int]
  jtdaj_nrow: wp.array[int]
  jtdaj_nblock: wp.array[int]
  J_rownnz: wp.array[int]
  J_rowadr: wp.array[int]
  J_colind: wp.array[int]
  J: wp.array[float]
  pos: wp.array[float]
  margin: wp.array[float]
  D: wp.array[float]
  vel: wp.array[float]
  aref: wp.array[float]
  frictionloss: wp.array[float]
  force: wp.array[float]
  state: wp.array[int]
  island: wp.array[int]
  Ma: wp.array[float]
  Jqvel: wp.array[float]

@dataclasses.dataclass
class Data:
  """Dynamic state that updates each step."""

  solver_niter: wp.array[int]
  ne: wp.array[int]
  nf: wp.array[int]
  nl: wp.array[int]
  nefc: wp.array[int]
  nisland: wp.array[int]
  nidof: wp.array[int]
  ntree_awake: wp.array[int]
  nbody_awake: wp.array[int]
  nv_awake: wp.array[int]
  time: wp.array[float]
  energy: wp.array[wp.vec2]
  qpos: wp.array[float]
  qvel: wp.array[float]
  act: wp.array[float]
  history: wp.array[float]
  plastic: wp.array[float]
  qacc_warmstart: wp.array[float]
  ctrl: wp.array[float]
  qfrc_applied: wp.array[float]
  xfrc_applied: wp.array[wp.spatial_vectorf]
  eq_active: wp.array[bool]
  mocap_pos: wp.array[wp.vec3]
  mocap_quat: wp.array[wp.quat]
  qacc: wp.array[float]
  act_dot: wp.array[float]
  userdata: wp.array[float]
  sensordata: wp.array[float]
  tree_asleep: wp.array[int]
  xpos: wp.array[wp.vec3]
  xquat: wp.array[wp.quat]
  xmat: wp.array[wp.mat33]
  xipos: wp.array[wp.vec3]
  ximat: wp.array[wp.mat33]
  xanchor: wp.array[wp.vec3]
  xaxis: wp.array[wp.vec3]
  geom_xpos: wp.array[wp.vec3]
  geom_xmat: wp.array[wp.mat33]
  site_xpos: wp.array[wp.vec3]
  site_xmat: wp.array[wp.mat33]
  cam_xpos: wp.array[wp.vec3]
  cam_xmat: wp.array[wp.mat33]
  light_xpos: wp.array[wp.vec3]
  light_xdir: wp.array[wp.vec3]
  subtree_com: wp.array[wp.vec3]
  cdof: wp.array[wp.spatial_vectorf]
  cinert: wp.array[vec10]
  flexvert_xpos: wp.array[wp.vec3]
  flex_hessian_valid: wp.array[bool]
  flexvert_hessian: wp.array[vec6]
  flexedge_hessian: wp.array[wp.mat33]
  flexedge_J: wp.array[float]
  flexedge_length: wp.array[float]
  ten_wrapadr: wp.array[int]
  ten_wrapnum: wp.array[int]
  ten_J: wp.array[float]
  ten_length: wp.array[float]
  wrap_obj: wp.array[wp.vec2i]
  wrap_xpos: wp.array[wp.spatial_vectorf]
  actuator_length: wp.array[float]
  moment_rownnz: wp.array[int]
  moment_rowadr: wp.array[int]
  moment_colind: wp.array[int]
  actuator_moment: wp.array[float]
  crb: wp.array[vec10]
  M: wp.array[float]
  qLD: wp.array[float]
  qLDiagInv: wp.array[float]
  tree_awake: wp.array[int]
  body_awake: wp.array[int]
  body_awake_ind: wp.array[int]
  dof_awake_ind: wp.array[int]
  flexedge_velocity: wp.array[float]
  ten_velocity: wp.array[float]
  actuator_velocity: wp.array[float]
  cvel: wp.array[wp.spatial_vectorf]
  cdof_dot: wp.array[wp.spatial_vectorf]
  qfrc_bias: wp.array[float]
  qfrc_spring: wp.array[float]
  qfrc_damper: wp.array[float]
  qfrc_gravcomp: wp.array[float]
  qfrc_fluid: wp.array[float]
  qfrc_adhesion: wp.array[float]
  qfrc_passive: wp.array[float]
  subtree_linvel: wp.array[wp.vec3]
  subtree_angmom: wp.array[wp.vec3]
  qH: wp.array[float]
  qHDiagInv: wp.array[float]
  qLU: wp.array[float]
  actuator_force: wp.array[float]
  qfrc_actuator: wp.array[float]
  qfrc_smooth: wp.array[float]
  qacc_smooth: wp.array[float]
  qfrc_constraint: wp.array[float]
  qfrc_inverse: wp.array[float]
  cacc: wp.array[wp.spatial_vectorf]
  cfrc_int: wp.array[wp.spatial_vectorf]
  cfrc_ext: wp.array[wp.spatial_vectorf]
  efm_c: wp.array[float]
  efm_diag: wp.array[float]
  efm_fluid: wp.array[float]
  efm_ca: wp.array[float]
  efm_K_val: wp.array[float]
  qHLD: wp.array[float]
  efm_ts: wp.array[float]
  efm_as: wp.array[float]
  contact: Contact
  efc: Constraint
  dof_frictionadr: wp.array[int]
  tree_island: wp.array[int]
  dof_island: wp.array[int]
  island_dofadr: wp.array[int]
  island_idofadr: wp.array[int]
  island_nv: wp.array[int]
  island_nefc: wp.array[int]
  island_ne: wp.array[int]
  island_nf: wp.array[int]
  island_iefcadr: wp.array[int]
  map_dof2idof: wp.array[int]
  map_idof2dof: wp.array[int]
  map_efc2iefc: wp.array[int]
  map_iefc2efc: wp.array[int]
  dof_islandid: wp.array[int]
  efc_islandid: wp.array[int]
  ncdof: wp.array[int]
  dof_cdof: wp.array[int]
  cdof_dof: wp.array[int]
  cdof_tri_row: wp.array[int]
  cdof_tri_col: wp.array[int]
  cM: wp.array[float]
  cqLD: wp.array[float]
  crhs: wp.array[float]
  cx: wp.array[float]
  cJ: wp.array[float]
  cMa: wp.array[float]
  cqfrc_smooth: wp.array[float]
  cqacc_smooth: wp.array[float]
  cqacc_warmstart: wp.array[float]
  cqacc: wp.array[float]
  cqfrc_constraint: wp.array[float]
  nworld: int
  naconmax: int
  naccdmax: int
  njmax: int
  nvmax: int
  nvmax_pad: int
  njmax_pad: int
  njmax_nnz: int
  nacon: wp.array[int]
  ncollision: wp.array[int]
  flex_aabb_min: wp.array[wp.vec3]
  flex_aabb_max: wp.array[wp.vec3]
  flexnode_xpos: wp.array[wp.vec3]
  overflow: wp.array[int]
  face_xpos: wp.array[wp.vec3]
  face_quat: wp.array[wp.quat]
  flex_rod_chain_ok: wp.array[bool]
  flex_rodhinge_g: wp.array[wp.vec3]
  flex_rodhinge_Wk: wp.array[wp.mat33]
  flex_rodhinge_Wd: wp.array[wp.mat33]
  flex_rodedge_ktheta: wp.array[float]
  flex_rodedge_grad: wp.array[float]
  flex_rodedge_frc: wp.array[wp.vec3]
  flex_rodedge_tridiag: wp.array[wp.vec2]
  flex_rod_end_torque: wp.array[wp.spatial_vectorf]
  flex_rod_chain_valid: wp.array[bool]
  flex_rod_ref_ok: wp.array[bool]
  flex_rod_frame0_loc: wp.array[wp.vec3]
  flex_rod_frame0_ref: wp.array[wp.vec3]
  flex_rod_tail_loc: wp.array[wp.vec3]
  flex_rodhinge_om0: wp.array[wp.vec2]

@dataclasses.dataclass
class InverseContext:
  """Workspace arrays for inverse dynamics."""

  Jaref: wp.array[float]
  search_dot: wp.array[float]
  done: wp.array[bool]
  quad_changed_ids: wp.array[int]
  quad_changed_count: wp.array[int]
  state_changed_count: wp.array[int]
  ls_exhausted: wp.array[bool]
  compact_m_full: Model | None = ...
  compact_d_full: Data | None = ...
  efm_con: (
    tuple[
      wp.array[int],
      wp.array[float],
      wp.array[float],
      wp.array[float],
      wp.array[int],
    ]
    | None
  ) = ...
  flex_scratch: tuple[wp.array[wp.vec3], wp.array[wp.vec3]] | None = ...
  tw_diag: wp.array[float] | None = ...

@dataclasses.dataclass
class SolverContext:
  """Workspace arrays for constraint solver."""

  Jaref: wp.array[float]
  search_dot: wp.array[float]
  done: wp.array[bool]
  grad: wp.array[float]
  grad_dot: wp.array[float]
  newton_decrement: wp.array[float]
  Mgrad: wp.array[float]
  search: wp.array[float]
  mv: wp.array[float]
  jv: wp.array[float]
  quad: wp.array[wp.vec3]
  alpha: wp.array[float]
  grad_scale: wp.array[float]
  state_changed_count: wp.array[int]
  improvement: wp.array[float]
  ls_exhausted: wp.array[bool]
  search_unchanged: wp.array[bool]
  prev_grad: wp.array[float]
  prev_Mgrad: wp.array[float]
  beta: wp.array[float]
  h: wp.array[float]
  hfactor: wp.array[float]
  quad_changed_ids: wp.array[int]
  quad_changed_count: wp.array[int]
  inertia_scale: wp.array[float]
  compact_m_full: Model | None = ...
  compact_d_full: Data | None = ...
  efm_con: (
    tuple[
      wp.array[int],
      wp.array[float],
      wp.array[float],
      wp.array[float],
      wp.array[int],
    ]
    | None
  ) = ...
  flex_scratch: tuple[wp.array[wp.vec3], wp.array[wp.vec3]] | None = ...
  solve_ld_scratch: wp.array[float] | None = ...
  tw_diag: wp.array[float] | None = ...
  h_tile_nz: wp.array[int] | None = ...
  h_row_mask: wp.array[wp.uint64] | None = ...
  h_col_mask: wp.array[wp.uint64] | None = ...
  h_tile_is_diag: wp.array[int] | None = ...
  h_rhs_work: wp.array[float] | None = ...

@dataclasses.dataclass
class RenderContext:
  """Context for rendering."""

  nrender: int
  cam_res: wp.array[wp.vec2i]
  cam_id_map: wp.array[int]
  use_textures: bool
  use_fast_math: bool
  use_shadows: bool
  use_ambient_lighting: bool
  background_color: wp.uint32
  use_precomputed_rays: bool
  render_skybox: bool
  skybox_tex_id: wp.array[int]
  skybox_face_width: wp.array[int]
  headlight_active: bool
  headlight_ambient: wp.vec3
  headlight_diffuse: wp.vec3
  headlight_specular: wp.vec3
  bvh_ngeom: int
  enabled_geom_ids: wp.array[int]
  mesh_registry: dict[int, wp.Mesh]
  mesh_bvh_id: wp.array[wp.uint64]
  mesh_bounds_size: wp.array[wp.vec3]
  mesh_texcoord: wp.array[wp.vec2]
  mesh_texcoord_offsets: wp.array[int]
  mesh_facetexcoord: wp.array[wp.vec3i]
  mesh_facenormal: wp.array[wp.vec3i]
  samples_per_pixel: int
  aa_accum: wp.array[wp.vec3]
  textures: wp.array[wp.Texture2D]
  textures_registry: list[wp.Texture2D]
  hfield_registry: dict[int, wp.Mesh]
  hfield_bvh_id: wp.array[wp.uint64]
  hfield_bounds_size: wp.array[wp.vec3]
  flex_mesh_registry: dict[int, wp.Mesh]
  flex_rgba: wp.array[wp.vec4]
  flex_bvh_id: wp.array[wp.uint64]
  flex_group_root: wp.array[int]
  flex_render_smooth: bool
  bvh_nflexgeom: int
  flex_dim_np: wp.array[int]
  flex_geom_flexid: wp.array[int]
  flex_geom_edgeid: wp.array[int]
  bvh: wp.Bvh
  bvh_id: wp.uint64
  lower: wp.array[wp.vec3]
  upper: wp.array[wp.vec3]
  group: wp.array[int]
  group_root: wp.array[int]
  ray: wp.array[wp.vec3]
  ray_offset: wp.array[wp.vec3]
  rgb_data: wp.array[wp.uint32]
  rgb_adr: wp.array[int]
  depth_data: wp.array[wp.float32]
  depth_adr: wp.array[int]
  render_rgb: wp.array[bool]
  render_depth: wp.array[bool]
  seg_data: wp.array[wp.vec2i]
  seg_adr: wp.array[int]
  render_seg: wp.array[bool]
  znear: float
  zfar: float
  total_rays: int
  enable_backface_culling: bool
  shadow_light_fraction: float
  enable_vertex_normals: bool
  enable_specular: bool
  enable_emission: bool
  enable_per_light_ambient: bool
  light_attenuation_is_default: bool
  has_spot_lights: bool
  has_orthographic_camera: bool
  splat_position: wp.array[wp.vec3]
  splat_rotation: wp.array[wp.quat]
  splat_scale: wp.array[wp.vec3]
  splat_rgba: wp.array[wp.vec4]
  splat_bvh: wp.Bvh | None
  splat_lower: wp.array[wp.vec3]
  splat_upper: wp.array[wp.vec3]
  splat_bvh_id: wp.uint64
  splat_group_root: wp.array[int]
  splat_count: int
  geom_ray_types: tuple[int, ...] = ...
  _megakernel: wp.Kernel | None = ...
