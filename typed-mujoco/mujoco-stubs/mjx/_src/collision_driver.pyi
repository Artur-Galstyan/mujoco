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

"""Runs collision checking for all geoms in a Model."""

from collections.abc import Iterator

import numpy as np
from numpy import typing as npt

import mujoco
from mujoco.mjx._src.types import Data, GeomType, Impl, Model

def has_collision_fn(t1: GeomType, t2: GeomType) -> bool:
  """Returns True if a collision function exists for a pair of geom types."""

def geom_pairs(
  m: Model | mujoco.MjModel,
) -> Iterator[tuple[int, int, int]]:
  """Yields geom pairs to check for collisions."""

def make_condim(
  m: Model | mujoco.MjModel, impl: Impl = ...
) -> npt.NDArray[np.int32]:
  """Returns the dims of the contacts for a Model."""

def collision(m: Model, d: Data) -> Data:
  """Collides geometries."""
