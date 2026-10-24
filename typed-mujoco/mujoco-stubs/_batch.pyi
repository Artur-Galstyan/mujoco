"""Native batch operations; array arguments must be C-contiguous and exact dtype."""

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
from numpy import typing as npt

from mujoco import MjData, MjModel, mjtStage, mjtState

_NumArray: TypeAlias = npt.NDArray[np.float32] | npt.NDArray[np.float64]
_FieldArray: TypeAlias = (
  _NumArray
  | npt.NDArray[np.int32]
  | npt.NDArray[np.uint8]
  | npt.NDArray[np.bool_]
  | npt.NDArray[np.int8]
  | npt.NDArray[np.uint64]
)
_Ids: TypeAlias = npt.NDArray[np.int32] | None

class Batch:
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
  def state(self) -> _NumArray: ...
  def state_component(self, component: int | mjtState) -> tuple[int, int]: ...
  def warning(self) -> npt.NDArray[np.int32]: ...
  def status(self) -> npt.NDArray[np.int32]: ...
  def error(self, sim: int) -> str: ...
  def output(self, name: str) -> npt.NDArray[np.uint8] | None: ...
  def expand(self, name: str) -> npt.NDArray[np.uint8] | None: ...
  def expanded(self, name: str) -> npt.NDArray[np.uint8] | None: ...
  def step(
    self, ids: _Ids = None, nstep: int = 1, stage: int | mjtStage = 0
  ) -> int: ...
  def forward(self, ids: _Ids = None, stage: int | mjtStage = 3) -> int: ...
  def reset(
    self,
    ids: _Ids = None,
    keyframe: npt.NDArray[np.int32] | None = None,
    stage: int | mjtStage = 3,
  ) -> int: ...
  def get_model(self, sim: int, model: MjModel) -> None: ...
  def get_data(self, sim: int, data: MjData) -> None: ...
  def set_models(self, models: MjModel | Sequence[MjModel] | None) -> None: ...
  def set_const(self, ids: _Ids = None) -> int: ...
  def rollout(
    self,
    ids: _Ids,
    nstep: int,
    control: _NumArray | None,
    control_spec: int | mjtState,
    records: Sequence[tuple[str | None, int | mjtState, _FieldArray]],
  ) -> int: ...
  def jac(
    self,
    ids: _Ids,
    body: npt.NDArray[np.int32],
    point: _NumArray,
    jacp: _NumArray | None,
    jacr: _NumArray | None,
  ) -> int: ...
  def ray(
    self,
    ids: _Ids,
    pnt: _NumArray,
    vec: _NumArray,
    geomgroup: npt.NDArray[np.uint8] | None,
    flg_static: bool,
    bodyexclude: npt.NDArray[np.int32],
    dist: _NumArray,
    geomid: npt.NDArray[np.int32],
    normal: _NumArray | None,
  ) -> int: ...
