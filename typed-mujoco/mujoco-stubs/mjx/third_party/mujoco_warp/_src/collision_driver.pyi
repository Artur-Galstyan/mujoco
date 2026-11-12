# Copyright 2025 The Newton Developers
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
"""Collision pipeline driver: broadphase and narrowphase dispatch."""

import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src.collision_core import (
  CollisionContext,
)
from mujoco.mjx.third_party.mujoco_warp._src.types import (
  CollisionType,
  Data,
  GeomType,
  Model,
)

MJ_COLLISION_TABLE: dict[tuple[GeomType, GeomType], CollisionType]
MJ_FLEX_COLLISION_TABLE: dict[tuple[GeomType, GeomType], CollisionType]

def sap_broadphase(
  m: Model,
  d: Data,
  ctx: CollisionContext,
  awake_prev: wp.array[int] | None = ...,
) -> None: ...
def nxn_broadphase(
  m: Model,
  d: Data,
  ctx: CollisionContext,
  awake_prev: wp.array[int] | None = ...,
) -> None: ...
def collision(
  m: Model, d: Data, awake_prev: wp.array[int] | None = ...
) -> None: ...
