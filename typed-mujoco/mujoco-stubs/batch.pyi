"""Run many simulations of one model without a viewer.

The batch owns live NumPy state rows; a named view selects an object's columns.
Only state fields are writable through `bind`; `expand` provides writable model
fields. Neither the Python nor native Batch supports a context manager.

>>> import mujoco
>>> from mujoco import batch
>>> model = mujoco.MjModel.from_xml_string(
...   '<mujoco><worldbody><body><joint name="j" type="slide"/>'
...   '<geom size=".1"/></body></worldbody></mujoco>'
... )
>>> simulations = batch.Batch(model, nsim=2, nthread=1)
>>> simulations.joint('j').qpos[:, 0] = [0.0, 0.1]
>>> simulations.step()
>>> simulations.state.shape == (2, simulations.nstate)
True
>>> simulations.get_data(1).time > 0
True
"""

from collections.abc import Mapping, Sequence
from typing import Final, Generic, Literal, TypeAlias, TypeVar, overload

import numpy as np
from numpy import typing as npt

from mujoco import MjData, MjModel, mjtStage, mjtState

_NumArray: TypeAlias = npt.NDArray[np.float32] | npt.NDArray[np.float64]
_IntArray: TypeAlias = npt.NDArray[np.int32]
_Float32Array: TypeAlias = npt.NDArray[np.float32]
_UInt8Array: TypeAlias = npt.NDArray[np.uint8]
_IndexArray: TypeAlias = _IntArray | npt.NDArray[np.int64]
_FieldArray: TypeAlias = (
  _NumArray
  | _IntArray
  | npt.NDArray[np.uint8]
  | npt.NDArray[np.bool_]
  | npt.NDArray[np.int8]
  | npt.NDArray[np.uint64]
)
_FloatDataName: TypeAlias = Literal[
  'M',
  'act',
  'act_dot',
  'actuator_force',
  'actuator_length',
  'actuator_moment',
  'actuator_velocity',
  'bvh_aabb_dyn',
  'cacc',
  'cam_xmat',
  'cam_xpos',
  'cdof',
  'cdof_dot',
  'cfrc_ext',
  'cfrc_int',
  'cinert',
  'crb',
  'ctrl',
  'cvel',
  'energy',
  'eq_active',
  'flexedge_J',
  'flexedge_hessian',
  'flexedge_length',
  'flexedge_velocity',
  'flexelem_aabb',
  'flexelem_krot',
  'flexvert_J',
  'flexvert_hessian',
  'flexvert_lambda',
  'flexvert_length',
  'flexvert_xpos',
  'geom_xmat',
  'geom_xpos',
  'history',
  'light_xdir',
  'light_xpos',
  'mocap_pos',
  'mocap_quat',
  'plugin_state',
  'qDeriv',
  'qH',
  'qHDiagInv',
  'qLD',
  'qLDiagInv',
  'qLU',
  'qacc',
  'qacc_smooth',
  'qacc_warmstart',
  'qfrc_actuator',
  'qfrc_adhesion',
  'qfrc_applied',
  'qfrc_bias',
  'qfrc_constraint',
  'qfrc_damper',
  'qfrc_fluid',
  'qfrc_gravcomp',
  'qfrc_inverse',
  'qfrc_passive',
  'qfrc_smooth',
  'qfrc_spring',
  'qpos',
  'qvel',
  'sensordata',
  'site_xmat',
  'site_xpos',
  'solver_fwdinv',
  'subtree_angmom',
  'subtree_com',
  'subtree_linvel',
  'ten_J',
  'ten_length',
  'ten_velocity',
  'time',
  'userdata',
  'wrap_xpos',
  'xanchor',
  'xaxis',
  'xfrc_applied',
  'ximat',
  'xipos',
  'xmat',
  'xpos',
  'xquat',
]
_IntDataName: TypeAlias = Literal[
  'body_awake',
  'body_awake_ind',
  'dof_awake_ind',
  'flexvert_conage',
  'moment_colind',
  'moment_rowadr',
  'moment_rownnz',
  'nA',
  'nJ',
  'ncon',
  'ne',
  'nefc',
  'nf',
  'nidof',
  'nisland',
  'nl',
  'parent_awake_ind',
  'plugin',
  'solver_niter',
  'solver_nnz',
  'ten_wrapadr',
  'ten_wrapnum',
  'tree_asleep',
  'tree_awake',
  'wrap_obj',
]
_BoolDataName: TypeAlias = Literal['bvh_active', 'flex_hessian_valid']
Ids: TypeAlias = npt.ArrayLike | None
RecordItem: TypeAlias = str | int | mjtState
DATA_FIELDS: frozenset[str]
MODEL_FIELDS: frozenset[str]

class SimulationError(RuntimeError):
  """Some simulations failed; `ids` lists them and `Batch.error` gives details.

  Other simulations finish their call. Invalid inputs raise ValueError instead;
  a batch also rejects Python physics callbacks (use C callbacks instead).
  """

  ids: _IndexArray
  def __init__(
    self, ids: _IndexArray, message: str, ncall: int | None = None
  ) -> None: ...

class _Named:
  @property
  def id(self) -> int: ...
  @property
  def name(self) -> str: ...

_T = TypeVar('_T')

class _Writable(Generic[_T]):
  def __get__(
    self, instance: _Named, owner: type[_Named] | None = None
  ) -> _T: ...
  def __set__(self, instance: _Named, value: npt.ArrayLike) -> None: ...

class _Actuator(_Named):
  acc0: _Writable[_NumArray]
  actadr: Final[_IntArray]
  actnum: Final[_IntArray]
  actrange: _Writable[_NumArray]
  biasprm: _Writable[_NumArray]
  biastype: Final[_IntArray]
  cranklength: _Writable[_NumArray]
  ctrl: _Writable[_NumArray]
  ctrlrange: _Writable[_NumArray]
  dynprm: _Writable[_NumArray]
  dyntype: Final[_IntArray]
  force: Final[_NumArray]
  forcerange: _Writable[_NumArray]
  gainprm: _Writable[_NumArray]
  gaintype: Final[_IntArray]
  gear: _Writable[_NumArray]
  group: _Writable[_IntArray]
  length: Final[_NumArray]
  length0: _Writable[_NumArray]
  lengthrange: _Writable[_NumArray]
  trnid: Final[_IntArray]
  trntype: Final[_IntArray]
  user: _Writable[_NumArray]
  velocity: Final[_NumArray]

class _Body(_Named):
  cacc: Final[_NumArray]
  cfrc_ext: Final[_NumArray]
  cfrc_int: Final[_NumArray]
  cinert: Final[_NumArray]
  crb: Final[_NumArray]
  cvel: Final[_NumArray]
  dofadr: Final[_IntArray]
  dofnum: Final[_IntArray]
  geomadr: Final[_IntArray]
  geomnum: Final[_IntArray]
  inertia: _Writable[_NumArray]
  invweight0: _Writable[_NumArray]
  ipos: _Writable[_NumArray]
  iquat: _Writable[_NumArray]
  jntadr: Final[_IntArray]
  jntnum: Final[_IntArray]
  mass: _Writable[_NumArray]
  mocapid: Final[_IntArray]
  parentid: Final[_IntArray]
  pos: _Writable[_NumArray]
  quat: _Writable[_NumArray]
  rootid: Final[_IntArray]
  sameframe: _Writable[_UInt8Array]
  simple: _Writable[_UInt8Array]
  subtree_angmom: Final[_NumArray]
  subtree_com: Final[_NumArray]
  subtree_linvel: Final[_NumArray]
  subtreemass: _Writable[_NumArray]
  user: _Writable[_NumArray]
  weldid: Final[_IntArray]
  xfrc_applied: _Writable[_NumArray]
  ximat: Final[_NumArray]
  xipos: Final[_NumArray]
  xmat: Final[_NumArray]
  xpos: Final[_NumArray]
  xquat: Final[_NumArray]

class _Camera(_Named):
  bodyid: Final[_IntArray]
  fovy: _Writable[_NumArray]
  ipd: _Writable[_NumArray]
  mat0: _Writable[_NumArray]
  mode: _Writable[_IntArray]
  pos: _Writable[_NumArray]
  pos0: _Writable[_NumArray]
  poscom0: _Writable[_NumArray]
  quat: _Writable[_NumArray]
  targetbodyid: Final[_IntArray]
  user: _Writable[_NumArray]
  xmat: Final[_NumArray]
  xpos: Final[_NumArray]

class _Equality(_Named):
  data: _Writable[_NumArray]
  obj1id: Final[_IntArray]
  obj2id: Final[_IntArray]
  solimp: _Writable[_NumArray]
  solref: _Writable[_NumArray]
  type: Final[_IntArray]

class _Geom(_Named):
  bodyid: Final[_IntArray]
  conaffinity: _Writable[_IntArray]
  condim: _Writable[_IntArray]
  contype: Final[_IntArray]
  dataid: Final[_IntArray]
  friction: _Writable[_NumArray]
  gap: _Writable[_NumArray]
  group: _Writable[_IntArray]
  margin: _Writable[_NumArray]
  matid: Final[_IntArray]
  pos: _Writable[_NumArray]
  priority: _Writable[_IntArray]
  quat: _Writable[_NumArray]
  rbound: _Writable[_NumArray]
  rgba: _Writable[_Float32Array]
  sameframe: _Writable[_UInt8Array]
  size: _Writable[_NumArray]
  solimp: _Writable[_NumArray]
  solmix: _Writable[_NumArray]
  solref: _Writable[_NumArray]
  type: Final[_IntArray]
  user: _Writable[_NumArray]
  xmat: Final[_NumArray]
  xpos: Final[_NumArray]

class _Joint(_Named):
  M0: _Writable[_NumArray]
  Madr: Final[_IntArray]
  armature: _Writable[_NumArray]
  axis: _Writable[_NumArray]
  bodyid: Final[_IntArray]
  cdof: Final[_NumArray]
  cdof_dot: Final[_NumArray]
  damping: _Writable[_NumArray]
  dampingpoly: _Writable[_NumArray]
  dofadr: Final[_IntArray]
  frictionloss: _Writable[_NumArray]
  group: _Writable[_IntArray]
  invweight0: _Writable[_NumArray]
  jntid: Final[_IntArray]
  margin: _Writable[_NumArray]
  parentid: Final[_IntArray]
  pos: _Writable[_NumArray]
  qLDiagInv: Final[_NumArray]
  qacc: Final[_NumArray]
  qacc_smooth: Final[_NumArray]
  qacc_warmstart: _Writable[_NumArray]
  qfrc_actuator: Final[_NumArray]
  qfrc_applied: _Writable[_NumArray]
  qfrc_bias: Final[_NumArray]
  qfrc_constraint: Final[_NumArray]
  qfrc_inverse: Final[_NumArray]
  qfrc_passive: Final[_NumArray]
  qfrc_smooth: Final[_NumArray]
  qpos: _Writable[_NumArray]
  qpos0: _Writable[_NumArray]
  qpos_spring: _Writable[_NumArray]
  qposadr: Final[_IntArray]
  qvel: _Writable[_NumArray]
  range: _Writable[_NumArray]
  simplenum: Final[_IntArray]
  solimp: _Writable[_NumArray]
  solref: _Writable[_NumArray]
  stiffness: _Writable[_NumArray]
  stiffnesspoly: _Writable[_NumArray]
  type: Final[_IntArray]
  user: _Writable[_NumArray]
  xanchor: Final[_NumArray]
  xaxis: Final[_NumArray]

class _Keyframe(_Named):
  act: _Writable[_NumArray]
  ctrl: _Writable[_NumArray]
  mpos: _Writable[_NumArray]
  mquat: _Writable[_NumArray]
  qpos: _Writable[_NumArray]
  qvel: _Writable[_NumArray]
  time: _Writable[_NumArray]

class _Light(_Named):
  ambient: _Writable[_Float32Array]
  attenuation: _Writable[_Float32Array]
  bodyid: Final[_IntArray]
  cutoff: _Writable[_Float32Array]
  diffuse: _Writable[_Float32Array]
  dir: _Writable[_NumArray]
  dir0: _Writable[_NumArray]
  exponent: _Writable[_Float32Array]
  mode: _Writable[_IntArray]
  pos: _Writable[_NumArray]
  pos0: _Writable[_NumArray]
  poscom0: _Writable[_NumArray]
  specular: _Writable[_Float32Array]
  targetbodyid: Final[_IntArray]
  type: Final[_IntArray]
  xdir: Final[_NumArray]
  xpos: Final[_NumArray]

class _Material(_Named):
  emission: _Writable[_Float32Array]
  reflectance: _Writable[_Float32Array]
  rgba: _Writable[_Float32Array]
  shininess: _Writable[_Float32Array]
  specular: _Writable[_Float32Array]
  texid: Final[_IntArray]
  texrepeat: _Writable[_Float32Array]

class _Numeric(_Named):
  adr: Final[_IntArray]
  data: _Writable[_NumArray]
  size: _Writable[_IntArray]

class _Pair(_Named):
  dim: _Writable[_IntArray]
  friction: _Writable[_NumArray]
  gap: _Writable[_NumArray]
  geom1: _Writable[_IntArray]
  geom2: _Writable[_IntArray]
  margin: _Writable[_NumArray]
  signature: _Writable[_IntArray]
  solimp: _Writable[_NumArray]
  solref: _Writable[_NumArray]

class _Sensor(_Named):
  adr: Final[_IntArray]
  cutoff: _Writable[_NumArray]
  data: Final[_NumArray]
  datatype: Final[_IntArray]
  dim: _Writable[_IntArray]
  needstage: _Writable[_IntArray]
  noise: _Writable[_NumArray]
  objid: Final[_IntArray]
  objtype: Final[_IntArray]
  refid: Final[_IntArray]
  reftype: Final[_IntArray]
  type: Final[_IntArray]
  user: _Writable[_NumArray]

class _Site(_Named):
  bodyid: Final[_IntArray]
  group: _Writable[_IntArray]
  matid: Final[_IntArray]
  pos: _Writable[_NumArray]
  quat: _Writable[_NumArray]
  rgba: _Writable[_Float32Array]
  sameframe: _Writable[_UInt8Array]
  size: _Writable[_NumArray]
  type: Final[_IntArray]
  user: _Writable[_NumArray]
  xmat: Final[_NumArray]
  xpos: Final[_NumArray]

class _Tendon(_Named):
  J_colind: _Writable[_IntArray]
  J_rowadr: Final[_IntArray]
  J_rownnz: _Writable[_IntArray]
  length: Final[_NumArray]
  velocity: Final[_NumArray]
  wrapadr: Final[_IntArray]
  wrapnum: Final[_IntArray]
  _adr: Final[_IntArray]
  _damping: Final[_NumArray]
  _dampingpoly: Final[_NumArray]
  _frictionloss: Final[_NumArray]
  _group: Final[_IntArray]
  _invweight0: Final[_NumArray]
  _length0: Final[_NumArray]
  _lengthspring: Final[_NumArray]
  _margin: Final[_NumArray]
  _matid: Final[_IntArray]
  _num: Final[_IntArray]
  _range: Final[_NumArray]
  _rgba: Final[_Float32Array]
  _solimp_fri: Final[_NumArray]
  _solimp_lim: Final[_NumArray]
  _solref_fri: Final[_NumArray]
  _solref_lim: Final[_NumArray]
  _stiffness: Final[_NumArray]
  _stiffnesspoly: Final[_NumArray]
  _user: Final[_NumArray]
  _width: Final[_NumArray]

class Batch:
  """Simulate `nsim` states with `nthread` threads (zero: all CPUs).

  `state` has shape (nsim, nstate), `warning` has (nsim, mjNWARNING, 2),
  and `status` has (nsim,). A persistent batch keeps one MjData per state.
  Calls cannot run concurrently from multiple Python threads.
  """

  model: MjModel
  state: _NumArray
  warning: _IntArray
  status: _IntArray

  def __init__(
    self, model: MjModel, nsim: int, nthread: int = 0, persistent: bool = False
  ) -> None: ...
  @property
  def nsim(self) -> int: ...
  @property
  def nthread(self) -> int: ...
  @property
  def nstate(self) -> int: ...
  @property
  def persistent(self) -> bool: ...
  @property
  def qpos(self) -> _NumArray: ...
  @qpos.setter
  def qpos(self, value: npt.ArrayLike) -> None: ...
  @property
  def ctrl(self) -> _NumArray: ...
  @ctrl.setter
  def ctrl(self, value: npt.ArrayLike) -> None: ...
  @overload
  def __getattr__(self, name: _FloatDataName) -> _NumArray: ...
  @overload
  def __getattr__(self, name: _IntDataName) -> _IntArray: ...
  @overload
  def __getattr__(self, name: _BoolDataName) -> npt.NDArray[np.bool_]: ...
  @overload
  def __getattr__(
    self, name: Literal['plugin_data']
  ) -> npt.NDArray[np.uint64]: ...
  def error(self, sim: int) -> str: ...
  @overload
  def bind(self, name: _FloatDataName) -> _NumArray: ...
  @overload
  def bind(self, name: _IntDataName) -> _IntArray: ...
  @overload
  def bind(self, name: _BoolDataName) -> npt.NDArray[np.bool_]: ...
  @overload
  def bind(self, name: Literal['plugin_data']) -> npt.NDArray[np.uint64]: ...
  @overload
  def bind(self, name: str) -> _FieldArray: ...
  def actuator(self, key: int | str) -> _Actuator: ...
  def body(self, key: int | str) -> _Body: ...
  def camera(self, key: int | str) -> _Camera: ...
  def equality(self, key: int | str) -> _Equality: ...
  def geom(self, key: int | str) -> _Geom: ...
  def joint(self, key: int | str) -> _Joint: ...
  def keyframe(self, key: int | str) -> _Keyframe: ...
  def light(self, key: int | str) -> _Light: ...
  def material(self, key: int | str) -> _Material: ...
  def numeric(self, key: int | str) -> _Numeric: ...
  def pair(self, key: int | str) -> _Pair: ...
  def sensor(self, key: int | str) -> _Sensor: ...
  def site(self, key: int | str) -> _Site: ...
  def tendon(self, key: int | str) -> _Tendon: ...
  @overload
  def expand(
    self, name: Literal['body_mass', 'geom_size', 'opt.gravity']
  ) -> _NumArray: ...
  @overload
  def expand(self, name: Literal['geom_rgba']) -> npt.NDArray[np.float32]: ...
  @overload
  def expand(self, name: Literal['body_simple']) -> npt.NDArray[np.uint8]: ...
  @overload
  def expand(self, name: Literal['body_parentid']) -> _IntArray: ...
  @overload
  def expand(self, name: Literal['plugin_attr']) -> npt.NDArray[np.int8]: ...
  @overload
  def expand(
    self, name: Literal['actuator_actearly']
  ) -> npt.NDArray[np.bool_]: ...
  @overload
  def expand(self, name: str) -> _FieldArray: ...
  def step(
    self, ids: Ids = None, nstep: int = 1, stage: int | mjtStage = 0
  ) -> None: ...
  def forward(self, ids: Ids = None, stage: int | mjtStage = 3) -> None: ...
  def reset(
    self,
    ids: Ids = None,
    keyframe: npt.ArrayLike = -1,
    stage: int | mjtStage = 3,
  ) -> None: ...
  def get_model(self, sim: int, model: MjModel | None = None) -> MjModel: ...
  def set_models(self, models: MjModel | Sequence[MjModel] | None) -> None: ...
  def get_data(self, sim: int, data: MjData | None = None) -> MjData: ...
  def set_const(self, ids: Ids = None) -> None: ...
  @overload
  def rollout(
    self,
    nstep: int,
    control: npt.ArrayLike | None = None,
    control_spec: int | mjtState = 64,
    *,
    record: Sequence[Literal['plugin_data']],
    ids: Ids = None,
    out: Mapping[Literal['plugin_data'], npt.NDArray[np.uint64]] | None = None,
  ) -> dict[Literal['plugin_data'], npt.NDArray[np.uint64]]: ...
  @overload
  def rollout(
    self,
    nstep: int,
    control: npt.ArrayLike | None = None,
    control_spec: int | mjtState = 64,
    record: Sequence[RecordItem] = ('qpos',),
    ids: Ids = None,
    out: Mapping[RecordItem, _FieldArray] | None = None,
  ) -> dict[RecordItem, _FieldArray]: ...
  def jac(
    self, body: npt.ArrayLike, point: npt.ArrayLike, ids: Ids = None
  ) -> tuple[_NumArray, _NumArray]: ...
  @overload
  def ray(
    self,
    pnt: npt.ArrayLike,
    vec: npt.ArrayLike,
    geomgroup: npt.ArrayLike | None = None,
    flg_static: bool = True,
    bodyexclude: npt.ArrayLike = -1,
    ids: Ids = None,
    normal: Literal[False] = False,
  ) -> tuple[_NumArray, _IntArray]: ...
  @overload
  def ray(
    self,
    pnt: npt.ArrayLike,
    vec: npt.ArrayLike,
    geomgroup: npt.ArrayLike | None = None,
    flg_static: bool = True,
    bodyexclude: npt.ArrayLike = -1,
    ids: Ids = None,
    *,
    normal: Literal[True],
  ) -> tuple[_NumArray, _IntArray, _NumArray]: ...
  @overload
  def ray(
    self,
    pnt: npt.ArrayLike,
    vec: npt.ArrayLike,
    geomgroup: npt.ArrayLike | None = None,
    flg_static: bool = True,
    bodyexclude: npt.ArrayLike = -1,
    ids: Ids = None,
    normal: bool = False,
  ) -> tuple[_NumArray, _IntArray] | tuple[_NumArray, _IntArray, _NumArray]: ...
