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
"""Forward dynamics and step for MJX Warp; auto-generated upstream."""

from mujoco.mjx._src import types

def forward(m: types.Model, d: types.Data) -> types.Data: ...
def forward_vmap(
  unused_axis_size: int,
  is_batched: tuple[types.Model, types.Data],
  m: types.Model,
  d: types.Data,
) -> tuple[types.Data, types.Data]: ...
def step(m: types.Model, d: types.Data) -> types.Data: ...
def step_vmap(
  unused_axis_size: int,
  is_batched: tuple[types.Model, types.Data],
  m: types.Model,
  d: types.Data,
) -> tuple[types.Data, types.Data]: ...
