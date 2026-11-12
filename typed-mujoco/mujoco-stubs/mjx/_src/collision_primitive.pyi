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

"""Collision primitives."""

from collections.abc import Callable
from typing import Protocol

import jax

from mujoco.mjx._src.collision_types import Collision as Collision
from mujoco.mjx._src.collision_types import FunctionKey, GeomInfo
from mujoco.mjx._src.types import Data, Model

class _Collider(Protocol):
  ncon: int
  def __call__(
    self, m: Model, d: Data, _: FunctionKey, geom: jax.Array
  ) -> Collision: ...

def collider(
  ncon: int,
) -> Callable[[Callable[[GeomInfo, GeomInfo], Collision]], _Collider]:
  """Wraps collision functions for use by collision_driver."""

plane_sphere: _Collider
plane_capsule: _Collider
plane_ellipsoid: _Collider
plane_cylinder: _Collider
sphere_sphere: _Collider
sphere_capsule: _Collider
capsule_capsule: _Collider
