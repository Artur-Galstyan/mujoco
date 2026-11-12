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
"""Utilities for testing."""

import numpy as np

import mujoco
from mujoco.mjx._src.types import Data

def benchmark(
  m: mujoco.MjModel,
  nstep: int = ...,
  batch_size: int = ...,
  unroll_steps: int = ...,
  solver: str = ...,
  iterations: int = ...,
  ls_iterations: int = ...,
) -> tuple[float, float, int]: ...
def efc_order(m: mujoco.MjModel, d: mujoco.MjData, dx: Data) -> np.ndarray: ...
def p(pct: int) -> bool: ...
def create_mjcf(
  seed: int,
  min_trees: int = ...,
  max_trees: int = ...,
  max_tree_depth: int = ...,
  body_pos: tuple[float, float, float] = ...,
  geom_pos: tuple[float, float, float] = ...,
  max_stacked_joints: int = ...,
  max_geoms_per_body: int = ...,
  max_contact_excludes: int = ...,
  max_contact_pairs: int = ...,
  disable_actuation_pct: int = ...,
  add_actuators: bool = ...,
  root_always_free: bool = ...,
  enable_contact: bool = ...,
) -> str: ...
def load_test_file(name: str) -> mujoco.MjModel: ...
