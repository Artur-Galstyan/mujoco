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

"""Convex collisions."""

from collections.abc import Callable
from typing import Protocol, TypeAlias

import jax

from mujoco.mjx._src.collision_types import Collision as Collision
from mujoco.mjx._src.collision_types import (
  ConvexInfo,
  FunctionKey,
  GeomInfo,
  HFieldInfo,
)
from mujoco.mjx._src.types import Data, Model

class _Collider(Protocol):
  ncon: int
  def __call__(
    self, m: Model, d: Data, key: FunctionKey, geom: jax.Array
  ) -> Collision: ...

_CollisionFn: TypeAlias = (
  Callable[[GeomInfo, ConvexInfo], Collision]
  | Callable[[ConvexInfo, ConvexInfo], Collision]
  | Callable[[HFieldInfo, GeomInfo, tuple[int, int]], Collision]
  | Callable[[HFieldInfo, ConvexInfo, tuple[int, int]], Collision]
)

def collider(ncon: int) -> Callable[[_CollisionFn], _Collider]:
  """Wraps collision functions for use by collision_driver."""

plane_convex: _Collider
sphere_convex: _Collider
capsule_convex: _Collider
box_box: _Collider
convex_convex: _Collider
hfield_sphere: _Collider
hfield_capsule: _Collider
hfield_convex: _Collider
