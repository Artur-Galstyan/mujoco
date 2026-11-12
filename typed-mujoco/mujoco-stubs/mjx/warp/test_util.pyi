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
"""Utilities for testing MJX MjWarp integration."""

from types import ModuleType

from numpy import typing as npt

import mujoco
from mujoco.mjx._src.types import Data

mjxw_forward: ModuleType | None

def assert_eq(a: npt.ArrayLike, b: npt.ArrayLike, name: str) -> None: ...
def assert_attr_eq(a: object, b: object, attr: str) -> None: ...
def make_data(
  m: mujoco.MjModel, worldid: int, naconmax: int = ..., njmax: int = ...
) -> Data: ...
def assert_contact_eq(d: mujoco.MjData, dx: Data, worldid: int) -> None: ...
def assert_efc_eq(d: mujoco.MjData, dx: Data, worldid: int) -> None: ...
def load_test_file(name: str) -> mujoco.MjModel: ...
