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

"""Optimization routines for system identification."""

from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np
import scipy.optimize as scipy_optimize
from numpy import typing as npt

from mujoco.sysid._src import parameter
from mujoco.sysid._src.timeseries import TimeSeries

def optimize(
  initial_params: parameter.ParameterDict,
  residual_fn: Callable[
    ...,
    tuple[
      Sequence[npt.NDArray[np.float64]],
      Sequence[Sequence[TimeSeries]],
      Sequence[Sequence[TimeSeries]],
    ],
  ],
  optimizer: Literal['scipy', 'mujoco', 'scipy_parallel_fd'] = 'mujoco',
  verbose: bool = True,
  check_conditioning: bool = False,
  **optimizer_kwargs: Any,
) -> tuple[parameter.ParameterDict, scipy_optimize.OptimizeResult]:
  """Run nonlinear least-squares optimization on the residual."""

def calculate_intervals(
  residuals_star: Sequence[npt.NDArray[np.float64]],
  J: npt.NDArray[np.float64] | None,
  alpha: float = 0.05,
  lambda_zero_thresh: float = 1e-15,
  v_zero_thresh: float = 1e-8,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
  """Calculate confidence intervals from the Jacobian at the optimum."""
