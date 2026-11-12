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

"""Mesh processing."""

import jax

import mujoco
from mujoco.mjx._src.collision_types import ConvexInfo, GeomInfo, HFieldInfo
from mujoco.mjx._src.types import ConvexMesh, Model

def box(info: GeomInfo) -> ConvexInfo:
  """Creates a box with rectangular faces."""

def convex(m: mujoco.MjModel | Model, data_id: int) -> ConvexMesh:
  """Processes a mesh for use in convex collision algorithms."""

def hfield_prism(vert: jax.Array) -> ConvexInfo:
  """Builds a hfield prism."""

def hfield(m: mujoco.MjModel | Model, data_id: int) -> HFieldInfo: ...
