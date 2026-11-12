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
"""Smooth dynamics: kinematics, inertia, bias forces and transmission."""

import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src.types import (
  Data,
  Model,
)

def kinematics(m: Model, d: Data) -> None: ...
def flex(m: Model, d: Data) -> None: ...
def com_pos(m: Model, d: Data) -> None:
  """Computes subtree center of mass positions."""

def camlight(m: Model, d: Data) -> None:
  """Computes camera and light positions and orientations."""

def crb(m: Model, d: Data) -> None: ...
def tendon_armature(m: Model, d: Data) -> None:
  """Add tendon armature to M."""

def factor_m(m: Model, d: Data) -> None:
  """Factorization of inertia-like matrix M, assumed spd."""

def rne(m: Model, d: Data, flg_acc: bool = ...) -> None:
  """Computes inverse dynamics using the recursive Newton-Euler algorithm."""

@wp.func
def transform_force(
  force: wp.vec3, torque: wp.vec3, offset: wp.vec3
) -> wp.spatial_vectorf: ...
def rne_postconstraint(m: Model, d: Data) -> None: ...
def tendon_bias(m: Model, d: Data, qfrc: wp.array[float]) -> None:
  """Add bias force due to tendon armature."""

def com_vel(m: Model, d: Data) -> None: ...
def transmission(m: Model, d: Data) -> None:
  """Computes actuator/transmission lengths and moments."""

def solve_LD(
  m: Model,
  d: Data,
  L: wp.array[float],
  D: wp.array[float],
  x: wp.array[float],
  y: wp.array[float],
  x_scratch: wp.array[float] | None = ...,
) -> None:
  """Computes backsubstitution for the inertia factorization."""

def skip_qacc_smooth(m: Model, d: Data) -> bool: ...
def solve_m(
  m: Model,
  d: Data,
  x: wp.array[float],
  y: wp.array[float],
  x_scratch: wp.array[float] | None = ...,
) -> None:
  """Computes backsubstitution: x = qLD * y."""

def factor_solve_i(
  m: Model,
  d: Data,
  M: wp.array[float],
  L: wp.array[float],
  D: wp.array[float],
  x: wp.array[float],
  y: wp.array[float],
) -> None:
  """Factorizes and solves the inertia-like linear system."""

def factor_solve_lu(
  m: Model,
  d: Data,
  qLU: wp.array[float],
  qacc: wp.array[float],
  qfrc: wp.array[float],
) -> None:
  """Factorize and solve non-symmetric implicit system: qacc = A \\ qfrc."""

def subtree_vel(m: Model, d: Data) -> None:
  """Computes subtree linear velocity and angular momentum."""

def tendon(m: Model, d: Data) -> None:
  """Computes tendon lengths and moments."""
