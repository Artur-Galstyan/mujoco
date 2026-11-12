# Copyright 2026 The Newton Developers
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
"""Flex collision detection (geom vs flex triangles)."""

import dataclasses

import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src.collision_core import (
  CollisionContext,
)
from mujoco.mjx.third_party.mujoco_warp._src.types import (
  Data,
  Model,
)

ENABLE_SAT_PREFILTER: bool

def flex_broadphase_aabb(m: Model, d: Data) -> None: ...

@dataclasses.dataclass
class FlexWorkspace:
  dist: wp.array[float]
  pos: wp.array[wp.vec3]
  nrm: wp.array[wp.vec3]
  geom: wp.array[wp.vec2i]
  flex: wp.array[wp.vec2i]
  elem: wp.array[wp.vec2i]
  vert: wp.array[wp.vec2i]
  worldid: wp.array[int]
  ncand: wp.array[int]
  filter_key: wp.array[wp.int64]
  filter_val: wp.array[int]
  cand_active: wp.array[int]
  epa_vert: wp.array[wp.vec3]
  epa_vert_index: wp.array[int]
  epa_face: wp.array[int]
  epa_pr: wp.array[wp.vec3]
  epa_norm2: wp.array[float]
  epa_horizon: wp.array[int]
  cand_active_sorted: wp.array[int] | None = ...
  flex_group_temp: wp.array[int] | None = ...
  flex_group_ids: wp.array[int] | None = ...
  flex_group_start_indices: wp.array[int] | None = ...
  flex_fps_min_dist: wp.array[float] | None = ...
  flex_num_groups: wp.array[int] | None = ...
  fps_scratch_dist: wp.array[float] | None = ...
  fps_scratch_cidx: wp.array[int] | None = ...
  fps_scratch_count: wp.array[int] | None = ...
  fps_selected_cidx: wp.array[int] | None = ...
  fps_selected_pos: wp.array[wp.vec3] | None = ...
  fps_groups_active: wp.array[int] | None = ...
  fps_condition: wp.array[int] | None = ...
  fps_iter: wp.array[int] | None = ...
  nccd: wp.array[int] | None = ...

def flex_collision(
  m: Model, d: Data, ctx: CollisionContext, enable_sat: bool | None = ...
) -> None: ...
