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
"""Bounding volume hierarchy construction and refit for rendering."""

from typing import TypeAlias

import numpy as np
import numpy.typing as npt
import warp as wp

import mujoco
from mujoco.mjx.third_party.mujoco_warp._src.types import (
  Data,
  Model,
  RenderContext,
)

_FloatArray: TypeAlias = npt.NDArray[np.floating]
_IntArray: TypeAlias = npt.NDArray[np.integer]

SPLAT_MIN_RESPONSE: float

def refit_bvh(m: Model, d: Data, rc: RenderContext) -> None: ...

compute_bvh_group_roots: wp.Kernel
compute_mesh_group_roots: wp.Kernel

def build_scene_bvh(
  mjm: mujoco.MjModel, mjd: mujoco.MjData, rc: RenderContext, nworld: int
) -> None: ...
def refit_scene_bvh(m: Model, d: Data, rc: RenderContext) -> None: ...
def build_mesh_bvh(
  mjm: mujoco.MjModel, meshid: int, constructor: str = ..., leaf_size: int = ...
) -> tuple[wp.Mesh, wp.vec3]: ...
def build_hfield_bvh(
  mjm: mujoco.MjModel,
  hfieldid: int,
  constructor: str = ...,
  leaf_size: int = ...,
) -> tuple[wp.Mesh, wp.vec3]: ...
def accumulate_flex_vertex_normals(deterministic: bool = ...) -> wp.Kernel: ...

normalize_vertex_normals: wp.Kernel

def build_flex_bvh(
  mjm: mujoco.MjModel,
  mjd: mujoco.MjData,
  nworld: int,
  flex_id: int,
  constructor: str = ...,
  leaf_size: int = ...,
  deterministic: bool = ...,
) -> tuple[wp.Mesh, wp.array[int]]: ...
def refit_flex_bvh(m: Model, d: Data, rc: RenderContext) -> None: ...
def build_splat_bvh(
  splat_position: _FloatArray,
  splat_rotation: _FloatArray,
  splat_scale: _FloatArray,
  splat_rgba: _FloatArray,
  splat_adr: _IntArray,
  splat_group_id: _IntArray,
  constructor: str = ...,
) -> tuple[
  wp.array[wp.vec3],
  wp.array[wp.quat],
  wp.array[wp.vec3],
  wp.array[wp.vec4],
  wp.Bvh,
  wp.uint64,
  wp.array[wp.vec3],
  wp.array[wp.vec3],
  wp.array[int],
  int,
]: ...
def refit_splat_bvh(rc: RenderContext) -> None: ...
