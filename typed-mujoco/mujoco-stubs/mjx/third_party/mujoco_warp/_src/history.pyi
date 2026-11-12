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

import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src.types import Data, Model

def reset_history(
  m: Model, d: Data, reset: wp.array[bool] | wp.array[int] | None = ...
) -> None: ...
def read_ctrl_delayed(m: Model, d: Data, ctrl: wp.array[float]) -> None: ...
def insert_ctrl_history(m: Model, d: Data) -> None: ...
def apply_sensor_delay(m: Model, d: Data, sensorid: wp.array[int]) -> None: ...
def read_ctrl(
  m: Model,
  d: Data,
  ctrlid: int,
  time: wp.array[float],
  interp: int,
  result: wp.array[float],
) -> None: ...
def read_sensor(
  m: Model,
  d: Data,
  sensorid: int,
  time: wp.array[float],
  interp: int,
  result: wp.array[float],
) -> None: ...
def init_ctrl_history(
  m: Model,
  d: Data,
  ctrlid: int,
  times: wp.array[float] | None,
  values: wp.array[float],
) -> None: ...
def init_sensor_history(
  m: Model,
  d: Data,
  sensorid: int,
  times: wp.array[float] | None,
  values: wp.array[float],
  phase: float | wp.array[float] | None = ...,
) -> None: ...
