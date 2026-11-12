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

from mujoco.mjx.third_party.mujoco_warp._src import types

def direct_dsu(
  m: types.Model, d: types.Data, island_parent: wp.array[int]
) -> None: ...
def island(m: types.Model, d: types.Data) -> None: ...
def compute_island_mapping(m: types.Model, d: types.Data) -> None: ...
def update_active_dofs(m: types.Model, d: types.Data) -> None: ...
