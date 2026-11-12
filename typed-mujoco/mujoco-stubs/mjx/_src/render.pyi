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

"""Render helpers for MJX."""

import jax

from mujoco.mjx._src.types import Data, Model
from mujoco.mjx.warp.render_context import RenderContextPytree

def render(
  m: Model, d: Data, ctx: RenderContextPytree
) -> tuple[jax.Array, jax.Array, Data]:
  """Render packed RGB and depth buffers."""

def render_with_segmentation(
  m: Model, d: Data, ctx: RenderContextPytree
) -> tuple[jax.Array, jax.Array, jax.Array, Data]:
  """Render and return RGB, depth, and packed segmentation outputs."""
