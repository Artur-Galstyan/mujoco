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

from mujoco.mjx.third_party.mujoco_warp._src import types

def set_const(
  m: types.Model,
  d: types.Data,
  groups: int = ...,
  restore: bool = ...,
) -> None: ...
def set_const_fixed(m: types.Model, d: types.Data) -> None: ...
def set_const_0(m: types.Model, d: types.Data, restore: bool = ...) -> None: ...
def set_length_range(
  m: types.Model, d: types.Data, index: int = ...
) -> None: ...
