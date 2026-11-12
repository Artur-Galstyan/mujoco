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

"""Collision base types."""

import dataclasses
from typing import Tuple, TypeAlias

import jax
import numpy as np
from numpy import typing as npt

from mujoco.mjx._src.dataclasses import PyTreeNode

Collision: TypeAlias = Tuple[jax.Array, jax.Array, jax.Array]

class GeomInfo(PyTreeNode):
  """Geom properties for primitive shapes."""

  pos: jax.Array
  mat: jax.Array
  size: jax.Array

class ConvexInfo(PyTreeNode):
  """Geom properties for convex meshes."""

  pos: jax.Array
  mat: jax.Array
  size: jax.Array
  vert: jax.Array
  face: jax.Array
  face_normal: jax.Array
  edge: jax.Array
  edge_face_normal: jax.Array

class HFieldInfo(PyTreeNode):
  """Geom properties for height fields."""

  pos: jax.Array
  mat: jax.Array
  size: npt.NDArray[np.float64]
  nrow: int
  ncol: int
  data: jax.Array

@dataclasses.dataclass(frozen=True)
class FunctionKey:
  """Specifies how geom pairs group into collision_driver's function table."""

  types: Tuple[int, int]
  data_ids: Tuple[int, int]
  condim: int
  subgrid_size: Tuple[int, int] = ...
