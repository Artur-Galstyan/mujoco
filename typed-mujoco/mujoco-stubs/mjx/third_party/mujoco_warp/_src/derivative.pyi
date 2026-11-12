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
"""Analytic derivatives of smooth forces and the effective metric."""

import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src.passive import (
  _EfmContact,
  _FlexScratch,
)
from mujoco.mjx.third_party.mujoco_warp._src.types import (
  Data,
  Model,
)

deriv_rne_cvel_cdof_dot: wp.Kernel

deriv_rne_cacc_cfrcbody_forward: wp.Kernel

def deriv_rne_cfrcbody_backward(deterministic: bool = ...) -> wp.Kernel: ...
def deriv_rne_body2jnt_sparse(deterministic: bool = ...) -> wp.Kernel: ...
def deriv_rne_vel(m: Model, d: Data, out: wp.array[float]) -> None:
  """Compute RNE velocity derivatives and add to the output."""

def deriv_smooth_vel(m: Model, d: Data, out: wp.array[float]) -> None:
  """Analytical derivative of smooth forces w.r.t. velocities."""

def eff_build(m: Model, d: Data) -> None:
  """Builds the implicit effective metric for the current position state."""

def eff_shift(m: Model, d: Data) -> None: ...
def eff_actuation(m: Model, d: Data) -> None:
  """Evaluates actuation-stage metric terms and factors the backbone qH."""

def eff_mul_m_scratch(
  m: Model, d: Data
) -> tuple[_EfmContact | None, _FlexScratch | None]:
  """Builds the eff_mul_m efm_con and flex_scratch arguments."""

def eff_mul_m(
  m: Model,
  d: Data,
  res: wp.array[float],
  vec: wp.array[float],
  skip: wp.array[bool] | None = ...,
  efm_con: _EfmContact | None = ...,
  flex_scratch: _FlexScratch | None = ...,
) -> None:
  """Multiplies vector by effective metric M_hat = M + h*D + h^2*K."""

def eff_prec_fold(
  m: Model, d: Data, tw_diag: wp.array[float] | None = ...
) -> None: ...
def eff_prec(
  m: Model,
  d: Data,
  res: wp.array[float],
  vec: wp.array[float],
  x_scratch: wp.array[float] | None = ...,
  tw_diag: wp.array[float] | None = ...,
) -> None: ...
def eff_solve(
  m: Model, d: Data, qacc: wp.array[float], qfrc: wp.array[float] | None = ...
) -> None:
  """Solves the effective metric system M_hat * qacc = qfrc_eff."""
