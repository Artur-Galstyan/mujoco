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
"""Forward dynamics, integrators and the simulation step."""

import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src.types import (
  Data,
  Model,
)

def check_discrete(m: Model) -> None:
  """Validate model configuration for discrete integrator."""

def euler(m: Model, d: Data) -> None:
  """Euler integrator, semi-implicit in velocity."""

def rungekutta4(m: Model, d: Data) -> None:
  """Runge-Kutta explicit order 4 integrator."""

def implicit_free_body(
  m: Model,
  d: Data,
  vec_in: wp.array[float],
  vec_out: wp.array[float],
  solve: bool,
) -> None: ...
def implicit_qlu(
  m: Model, d: Data, qLU: wp.array[float] | None = ...
) -> None: ...
def implicit(m: Model, d: Data) -> None:
  """Integrates fully implicit in velocity."""

def fwd_kinematics(m: Model, d: Data) -> None:
  """Kinematics-dependent computations."""

def fwd_position(m: Model, d: Data, factorize: bool = ...) -> None:
  """Position-dependent computations."""

def fwd_velocity(m: Model, d: Data) -> None:
  """Velocity-dependent computations."""

def fwd_actuation(m: Model, d: Data) -> None:
  """Actuation-dependent computations."""

def fwd_acceleration(m: Model, d: Data, factorize: bool = ...) -> None:
  """Add up all non-constraint forces, compute qacc_smooth."""

def forward(m: Model, d: Data) -> None:
  """Forward dynamics."""

def discrete(m: Model, d: Data) -> None:
  """Advance simulation using discrete integrator."""

def step(m: Model, d: Data) -> None:
  """Advance simulation."""

def step1(m: Model, d: Data) -> None:
  """Advance simulation in two phases: before input is set by user."""

def step2(m: Model, d: Data) -> None:
  """Advance simulation in two phases: after input is set by user."""
