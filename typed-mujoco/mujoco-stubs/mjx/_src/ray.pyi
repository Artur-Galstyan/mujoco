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

"""Functions for ray interesection testing."""

from collections.abc import Sequence

import jax

from mujoco.mjx._src.types import Data, GeomType, Model

def ray(
  m: Model,
  d: Data,
  pnt: jax.Array,
  vec: jax.Array,
  geomgroup: Sequence[int] = ...,
  flg_static: bool = ...,
  bodyexclude: Sequence[int] | int = ...,
) -> tuple[jax.Array, jax.Array]:
  """Returns the distance and geom id at which a ray intersects with a geom."""

def ray_geom(
  size: jax.Array, pnt: jax.Array, vec: jax.Array, geomtype: GeomType
) -> jax.Array:
  """Returns the distance at which a ray intersects with a primitive geom."""
