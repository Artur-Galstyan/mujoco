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
"""Convex geom narrowphase collision using GJK and EPA kernels."""

from collections.abc import Sequence
from typing import overload

import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src.collision_core import (
  CollisionContext,
)
from mujoco.mjx.third_party.mujoco_warp._src.types import (
  Data,
  GeomType,
  Model,
)

class vec_maxconpair:
  @overload
  def __init__(self) -> None: ...
  @overload
  def __init__(self, other: vec_maxconpair) -> None: ...
  @overload
  def __init__(self, *args: wp.float32 | float) -> None: ...
  @overload
  def __init__(self, args: Sequence[wp.float32]) -> None: ...

class mat_maxconpair:
  @overload
  def __init__(self) -> None: ...
  @overload
  def __init__(self, other: mat_maxconpair) -> None: ...
  @overload
  def __init__(self, *args: wp.float32 | float) -> None: ...
  @overload
  def __init__(self, *args: wp.vec3f) -> None: ...
  @overload
  def __init__(self, args: Sequence[wp.float32]) -> None: ...

def ccd_hfield_kernel_builder(
  geomtype1: int,
  geomtype2: int,
  gjk_iterations: int,
  epa_iterations: int,
  geomgeomid: int,
  warn_overflow: int,
) -> wp.Kernel: ...
def ccd_kernel_builder(
  geomtype1: int,
  geomtype2: int,
  gjk_iterations: int,
  epa_iterations: int,
  use_multiccd: bool,
  geomgeomid: int,
  block_dim: int,
  warn_overflow: int,
) -> wp.Kernel: ...
def convex_narrowphase(
  m: Model,
  d: Data,
  ctx: CollisionContext,
  collision_table: list[tuple[GeomType, GeomType]],
) -> None: ...
