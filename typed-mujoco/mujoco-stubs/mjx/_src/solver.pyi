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
"""Constraint solvers."""

import jax

from mujoco.mjx._src.dataclasses import PyTreeNode
from mujoco.mjx._src.types import (
  Data,
  Model,
)

class Context(PyTreeNode):
  qacc: jax.Array
  qfrc_constraint: jax.Array
  Jaref: jax.Array
  efc_force: jax.Array
  Ma: jax.Array
  grad: jax.Array
  Mgrad: jax.Array
  search: jax.Array
  gauss: jax.Array
  cost: jax.Array
  prev_cost: jax.Array
  solver_niter: jax.Array
  active: jax.Array
  fri: jax.Array
  dm: jax.Array
  u: jax.Array
  h: jax.Array

  @classmethod
  def create(cls, m: Model, d: Data, grad: bool = True) -> Context: ...

def solve(m: Model, d: Data) -> Data: ...
