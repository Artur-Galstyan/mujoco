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
"""MJX Warp render context types and buffer registry."""

from typing import Any

import warp as wp

from mujoco.mjx._src import dataclasses as mjx_dataclasses
from mujoco.mjx.third_party.mujoco_warp._src.types import (
  RenderContext as _WarpRenderContext,
)

class RenderContext:
  """MJX render context wrapping one or more warp render contexts."""

  key: int
  nworld: int
  rgb_data_shape: tuple[int, ...]
  depth_data_shape: tuple[int, ...]
  seg_data_shape: tuple[int, ...]
  seg_data_buffer: wp.array[wp.vec2i]
  def __init__(
    self,
    key: int,
    contexts: dict[int, _WarpRenderContext],
    default: _WarpRenderContext,
  ) -> None: ...
  def pytree(self) -> RenderContextPytree:
    """Returns a lightweight JAX pytree for use in jit/vmap."""

  def __getattr__(self, name: str) -> Any: ...
  def __del__(self) -> None: ...

class RenderContextPytree(mjx_dataclasses.PyTreeNode):
  """Minimal JAX pytree holding just the render context key."""

  key: int

def get(rc: RenderContextPytree) -> _WarpRenderContext:
  """Validates and returns the backing Warp render context."""
