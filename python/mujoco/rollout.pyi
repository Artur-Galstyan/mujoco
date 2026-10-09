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

from collections.abc import Sequence
from types import TracebackType
from typing import Literal, TypeAlias, overload

import numpy as np
from numpy import typing as npt
from typing_extensions import Self

from mujoco import MjData, MjModel

_NumArray: TypeAlias = npt.NDArray[np.float32] | npt.NDArray[np.float64]
_Model: TypeAlias = MjModel | Sequence[MjModel]
_Data: TypeAlias = MjData | Sequence[MjData]

class Rollout:
  nthread: int

  def __init__(self, *, nthread: int | None = None) -> None: ...
  def __enter__(self) -> Self: ...
  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc_val: BaseException | None,
    exc_tb: TracebackType | None,
  ) -> None: ...
  def close(self) -> None: ...
  @overload
  def rollout(
    self,
    model: _Model,
    data: _Data,
    initial_state: npt.ArrayLike,
    control: npt.ArrayLike | None = None,
    *,
    control_spec: int = ...,
    skip_checks: Literal[False] = False,
    nstep: int | None = None,
    initial_warmstart: npt.ArrayLike | None = None,
    state: npt.ArrayLike | None = None,
    sensordata: npt.ArrayLike | None = None,
    chunk_size: int | None = None,
  ) -> tuple[_NumArray, _NumArray]: ...
  @overload
  def rollout(
    self,
    model: _Model,
    data: _Data,
    initial_state: npt.ArrayLike,
    control: npt.ArrayLike | None = None,
    *,
    control_spec: int = ...,
    skip_checks: bool,
    nstep: int | None = None,
    initial_warmstart: npt.ArrayLike | None = None,
    state: npt.ArrayLike | None = None,
    sensordata: npt.ArrayLike | None = None,
    chunk_size: int | None = None,
  ) -> tuple[npt.ArrayLike | None, npt.ArrayLike | None]: ...

persistent_rollout: Rollout | None

def shutdown_persistent_pool() -> None: ...
@overload
def rollout(
  model: _Model,
  data: _Data,
  initial_state: npt.ArrayLike,
  control: npt.ArrayLike | None = None,
  *,
  control_spec: int = ...,
  skip_checks: Literal[False] = False,
  nstep: int | None = None,
  initial_warmstart: npt.ArrayLike | None = None,
  state: npt.ArrayLike | None = None,
  sensordata: npt.ArrayLike | None = None,
  chunk_size: int | None = None,
  persistent_pool: bool = False,
) -> tuple[_NumArray, _NumArray]: ...
@overload
def rollout(
  model: _Model,
  data: _Data,
  initial_state: npt.ArrayLike,
  control: npt.ArrayLike | None = None,
  *,
  control_spec: int = ...,
  skip_checks: bool,
  nstep: int | None = None,
  initial_warmstart: npt.ArrayLike | None = None,
  state: npt.ArrayLike | None = None,
  sensordata: npt.ArrayLike | None = None,
  chunk_size: int | None = None,
  persistent_pool: bool = False,
) -> tuple[npt.ArrayLike | None, npt.ArrayLike | None]: ...
