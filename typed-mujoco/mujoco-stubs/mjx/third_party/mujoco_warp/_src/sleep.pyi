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

K_AWAKE_VAL: int

def update_sleep(
  m: types.Model, d: types.Data, flg_staticawake: int = ...
) -> None: ...
def update_sleep_trees(m: types.Model, d: types.Data) -> None: ...
def wake(m: types.Model, d: types.Data) -> None: ...
def wake_collision(m: types.Model, d: types.Data) -> None: ...
def wake_tendon(m: types.Model, d: types.Data) -> None: ...
def wake_equality(m: types.Model, d: types.Data) -> None: ...
def sleep(m: types.Model, d: types.Data) -> None: ...
