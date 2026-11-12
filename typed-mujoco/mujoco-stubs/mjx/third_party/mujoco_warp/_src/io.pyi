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

from collections.abc import Mapping, Sequence
from typing import TypedDict

import numpy as np
import warp as wp
from numpy import typing as npt

import mujoco
from mujoco.mjx.third_party.mujoco_warp._src import types

CONTACT_EXCLUDE_INCLUDE: int
CONTACT_EXCLUDE_PASSIVE: int

class _MBlockLayout(TypedDict):
  total: int
  dof_adr: npt.NDArray[np.int32]
  scalar_tiles: dict[int, list[int]]
  gather_tiles: dict[int, list[int]]
  has_sparse: bool

def is_sparse(mjm: mujoco.MjModel) -> bool: ...
def m_block_layout(mjm: mujoco.MjModel) -> _MBlockLayout: ...
def put_model(
  mjm: mujoco.MjModel, batch_sizes: dict[str, int] | None = ...
) -> types.Model: ...
def make_data(
  mjm: mujoco.MjModel,
  nworld: int = ...,
  nconmax: int | None = ...,
  nccdmax: int | None = ...,
  njmax: int | None = ...,
  njmax_nnz: int | None = ...,
  naconmax: int | None = ...,
  naccdmax: int | None = ...,
  nvmax: int | None = ...,
) -> types.Data: ...
def put_data(
  mjm: mujoco.MjModel,
  mjd: mujoco.MjData,
  nworld: int = ...,
  nconmax: int | None = ...,
  nccdmax: int | None = ...,
  njmax: int | None = ...,
  njmax_nnz: int | None = ...,
  naconmax: int | None = ...,
  naccdmax: int | None = ...,
  nvmax: int | None = ...,
) -> types.Data: ...
def get_data_into(
  result: mujoco.MjData,
  mjm: mujoco.MjModel,
  d: types.Data,
  world_id: int = ...,
) -> None: ...
def reset_data(
  m: types.Model,
  d: types.Data,
  reset: wp.array[bool] | wp.array[int] | None = ...,
) -> None: ...
def reset_data_keyframe(
  m: types.Model, d: types.Data, key: int | wp.array[int]
) -> None: ...
def override_model(
  model: types.Model | mujoco.MjModel,
  overrides: Mapping[
    str, str | int | float | bool | Sequence[float] | npt.NDArray[np.generic]
  ]
  | Sequence[str],
) -> None: ...
def find_keys(model: mujoco.MjModel, keyname_prefix: str) -> list[int]: ...
def make_trajectory(
  model: mujoco.MjModel, keys: list[int]
) -> npt.NDArray[np.float64]: ...
def load_trajectory(
  npz_path: str, mjm: mujoco.MjModel, mjd: mujoco.MjData
) -> npt.NDArray[np.float32 | np.float64]: ...
