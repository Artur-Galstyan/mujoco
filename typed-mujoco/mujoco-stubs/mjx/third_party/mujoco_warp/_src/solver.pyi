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
"""Constraint solvers (CG, Newton) and no-slip post-processing."""

from mujoco.mjx.third_party.mujoco_warp._src.types import (
  Data,
  InverseContext,
  Model,
  SolverContext,
)

def create_inverse_context(m: Model, d: Data) -> InverseContext:
  """Create an InverseContext with allocated workspace arrays."""

def init_context(
  m: Model,
  d: Data,
  ctx: SolverContext | InverseContext,
  grad: bool = ...,
  compact: bool = ...,
) -> None: ...
def solve(m: Model, d: Data) -> None: ...
def smooth_solve_compact(m: Model, d: Data) -> None: ...
def solve_compact(m: Model, d: Data) -> None:
  """Run the dense Newton constraint solver in compacted DOF space."""

def noslip(m: Model, d: Data) -> None:
  """Noslip solver: projected Gauss-Seidel over the frictional constraints."""
