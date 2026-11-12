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
"""Functions to initialize, load, or save data."""

from collections.abc import Sequence
from typing import TypeAlias, TypedDict

import jax
import numpy as np
from numpy import typing as npt
from typing_extensions import Unpack

import mujoco
from mujoco.mjx._src import types
from mujoco.mjx.warp import render_context
from mujoco.mjx.warp import types as mjxw_types

_KeepaliveRefs: TypeAlias = dict[
  int, mujoco.MjModel | mujoco.MjData | list[mujoco.MjData]
]

class _RenderContextKwargs(TypedDict, total=False):
  cam_res: list[tuple[int, int]] | tuple[int, int] | None
  render_rgb: list[bool] | bool | None
  render_depth: list[bool] | bool | None
  render_seg: list[bool] | bool | None
  use_textures: bool
  use_fast_math: bool
  use_shadows: bool
  use_ambient_lighting: bool
  enabled_geom_groups: list[int]
  cam_active: list[bool] | list[str] | list[int] | None
  background_color: tuple[float, float, float, float]
  flex_render_smooth: bool
  use_precomputed_rays: bool
  render_skybox: bool
  enable_backface_culling: bool
  shadow_light_fraction: float
  samples_per_pixel: int
  enable_vertex_normals: bool
  enable_specular: bool
  enable_emission: bool
  enable_per_light_ambient: bool
  splat_position: npt.NDArray[np.floating] | None
  splat_rotation: npt.NDArray[np.floating] | None
  splat_scale: npt.NDArray[np.floating] | None
  splat_rgba: npt.NDArray[np.floating] | None
  splat_adr: npt.NDArray[np.integer] | None
  splat_group_id: npt.NDArray[np.integer] | None
  deterministic: bool

def has_cuda_gpu_device() -> bool: ...
def put_model(
  m: mujoco.MjModel,
  device: jax.Device | None = ...,
  impl: str | types.Impl | None = ...,
  graph_mode: mjxw_types.GraphMode | None = ...,
  keepalive_refs: _KeepaliveRefs | None = ...,
  batch_sizes: dict[str, int] | None = ...,
) -> types.Model: ...
def make_data(
  m: types.Model | mujoco.MjModel,
  device: jax.Device | None = ...,
  impl: str | types.Impl | None = ...,
  _full_compat: bool = ...,
  naconmax: int | None = ...,
  naccdmax: int | None = ...,
  njmax: int | None = ...,
  nvmax: int | None = ...,
  keepalive_refs: _KeepaliveRefs | None = ...,
) -> types.Data: ...
def put_data(
  m: mujoco.MjModel,
  d: mujoco.MjData,
  device: jax.Device | None = ...,
  impl: str | types.Impl | None = ...,
  naconmax: int | None = ...,
  naccdmax: int | None = ...,
  njmax: int | None = ...,
  nvmax: int | None = ...,
  dummy_arg_for_batching: jax.Array | None = ...,
  keepalive_refs: _KeepaliveRefs | None = ...,
) -> types.Data: ...
def get_data_into(
  result: mujoco.MjData | list[mujoco.MjData],
  m: mujoco.MjModel,
  d: types.Data,
  keepalive_refs: _KeepaliveRefs | None = ...,
) -> None: ...
def get_data(
  m: mujoco.MjModel,
  d: types.Data,
  keepalive_refs: _KeepaliveRefs | None = ...,
) -> mujoco.MjData | list[mujoco.MjData]: ...
def state_size(m: types.Model, spec: int | mujoco.mjtState) -> int: ...
def get_state(
  m: types.Model, d: types.Data, spec: int | mujoco.mjtState
) -> jax.Array: ...
def set_state(
  m: types.Model,
  d: types.Data,
  state: jax.Array,
  spec: int | mujoco.mjtState,
) -> types.Data: ...
def create_render_context(
  mjm: mujoco.MjModel,
  nworld: int,
  devices: Sequence[str] | None = ...,
  **kwargs: Unpack[_RenderContextKwargs],
) -> render_context.RenderContext: ...
