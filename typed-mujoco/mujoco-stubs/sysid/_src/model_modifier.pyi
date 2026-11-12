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

"""Model modifiers."""

from typing import Protocol

import numpy as np
from numpy import typing as npt

import mujoco
from mujoco.sysid._src.parameter import (
  InertiaType,
  ModifierFn,
  Parameter,
  ParameterDict,
)

class _PositionActuatorLike(Protocol):
  @property
  def gaintype(self) -> object: ...
  @property
  def biastype(self) -> object: ...
  @property
  def dyntype(self) -> object: ...

def remove_visuals(in_spec: mujoco.MjSpec) -> mujoco.MjSpec:
  """Remove visual elements from a Spec."""

def apply_param_modifiers_spec(
  params: ParameterDict, spec: mujoco.MjSpec
) -> mujoco.MjSpec: ...
def apply_param_modifiers(
  params: ParameterDict, spec: mujoco.MjSpec
) -> mujoco.MjModel: ...
def is_position_actuator(
  actuator: _PositionActuatorLike,
) -> bool:
  """Check if an actuator is a position actuator."""

def get_actuator_pd_gains(
  model: mujoco.MjModel, actuator_name: str
) -> tuple[float, float]:
  """Return the (P, D) gains of a position actuator."""

def apply_pgain(
  spec: mujoco.MjSpec,
  actuator_name: str,
  value: float | npt.NDArray[np.float64],
) -> mujoco.MjSpec:
  """Set the proportional gain for a position actuator."""

def apply_dgain(
  spec: mujoco.MjSpec,
  actuator_name: str,
  value: float | npt.NDArray[np.float64],
) -> mujoco.MjSpec:
  """Set the derivative gain for a position actuator."""

def apply_pdgain(
  spec: mujoco.MjSpec,
  actuator_name: str,
  value: npt.NDArray[np.float64],
) -> mujoco.MjSpec:
  """Set both proportional and derivative gains for a position actuator."""

def apply_body_mass_ipos(
  spec: mujoco.MjSpec,
  body_name: str,
  mass: float | npt.NDArray[np.float64] | None = None,
  ipos: npt.NDArray[np.float64] | None = None,
  rot_inertia_scale: bool = False,
) -> mujoco.MjSpec:
  """Apply mass and center-of-mass position to a body."""

def scale_body_inertia(
  spec: mujoco.MjSpec,
  body_name: str,
  value: float | npt.NDArray[np.float64],
) -> mujoco.MjSpec: ...
def pi_from_theta(
  theta: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
  """Convert base parameters θ to inertial parameters π."""

def pseudoinertia_from_pi(
  pi: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
  """Converts inertial parameters π to a 4x4 pseudoinertia matrix J."""

def cholesky_decompose_upper(
  J: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
  """Perform an upper-triangular Cholesky decomposition of J."""

def theta_from_pseudoinertia(
  J: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
  """Extract the 10-D vector of base parameters θ from the pseudoinertia J."""

def skew(v: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
  """Skew-symmetric matrix from a length-3 vector."""

def inertia_to_fullinertia(
  q: npt.NDArray[np.float64], inertia: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
  """Convert diagonal inertia with quaternion to full inertia matrix."""

def pi_from_body(
  spec: mujoco.MjSpec, body_name: str
) -> npt.NDArray[np.float64]:
  """Extracts the 10-D vector of inertial parameters π from a MuJoCo body."""

def theta_inertia_from_body(
  spec: mujoco.MjSpec, body_name: str
) -> npt.NDArray[np.float64]:
  """Extract base parameters θ from a body's inertial properties."""

def apply_body_theta_inertia(
  spec: mujoco.MjSpec,
  body_name: str,
  theta: npt.NDArray[np.float64],
) -> mujoco.MjSpec:
  """Apply base-parameter inertia θ to a body in the spec."""

def apply_body_inertia(
  spec: mujoco.MjSpec, name: str, param: Parameter
) -> None:
  """Apply inertia parameters to a body based on the parameter type."""

def body_inertia_param(
  spec: mujoco.MjSpec,
  model: mujoco.MjModel,
  body_name: str,
  inertia_type: InertiaType = ...,
  scale_rot_inertia: bool = False,
  mass_bound_mult: npt.NDArray[np.float64] | None = None,
  ipos_bound_off: npt.NDArray[np.float64] | None = None,
  stretch_bound_mult: npt.NDArray[np.float64] | None = None,
  shear_bound_off: npt.NDArray[np.float64] | None = None,
  param_name: str | None = None,
  modifier: ModifierFn | None = None,
) -> Parameter:
  """Creates Parameter objects for the inertia of a body in a simplified manner."""
