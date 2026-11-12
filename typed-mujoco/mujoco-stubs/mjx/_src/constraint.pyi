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
"""Core non-smooth constraint functions."""

import numpy as np
from numpy import typing as npt

import mujoco
from mujoco.mjx._src.types import (
  Data,
  Model,
)

def counts(
  efc_type: npt.NDArray[np.int_],
) -> tuple[np.int64, np.int64, np.int64, np.int64]: ...
def make_efc_type(
  m: Model | mujoco.MjModel, dim: npt.NDArray[np.int_] | None = None
) -> npt.NDArray[np.int_]: ...
def make_efc_address(
  m: Model | mujoco.MjModel,
  dim: npt.NDArray[np.int_],
  efc_type: npt.NDArray[np.int_],
) -> npt.NDArray[np.int_]: ...
def make_constraint(m: Model, d: Data) -> Data: ...
