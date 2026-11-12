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

"""Collision functions for shapes represented as signed distance functions."""

from collections.abc import Callable
from typing import TypeAlias

import jax

from mujoco.mjx._src.collision_primitive import _Collider
from mujoco.mjx._src.collision_types import Collision as Collision
from mujoco.mjx._src.collision_types import GeomInfo
from mujoco.mjx._src.dataclasses import PyTreeNode

SDFFn: TypeAlias = Callable[[jax.Array], jax.Array]

def collider(
  ncon: int,
) -> Callable[[Callable[[GeomInfo, GeomInfo], Collision]], _Collider]:
  """Wraps collision functions for use by collision_driver."""

def cylinder_jvp(
  primals: tuple[jax.Array, jax.Array],
  tangents: tuple[jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array]: ...

class GradientState(PyTreeNode):
  dist: jax.Array
  x: jax.Array

sphere_ellipsoid: _Collider
sphere_cylinder: _Collider
capsule_ellipsoid: _Collider
capsule_cylinder: _Collider
ellipsoid_ellipsoid: _Collider
ellipsoid_cylinder: _Collider
cylinder_cylinder: _Collider
