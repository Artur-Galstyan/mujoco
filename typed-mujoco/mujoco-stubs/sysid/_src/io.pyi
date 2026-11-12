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

"""I/O utilities for saving system identification results."""

import os
from collections.abc import Callable, Sequence

import numpy as np
import scipy.optimize as scipy_optimize
from numpy import typing as npt

from mujoco.sysid._src import parameter
from mujoco.sysid._src.timeseries import TimeSeries
from mujoco.sysid._src.trajectory import ModelSequences

def save_results(
  experiment_results_folder: str | os.PathLike[str],
  models_sequences: Sequence[ModelSequences],
  initial_params: parameter.ParameterDict,
  opt_params: parameter.ParameterDict,
  opt_result: scipy_optimize.OptimizeResult,
  residual_fn: Callable[
    ...,
    tuple[
      Sequence[npt.NDArray[np.float64]],
      Sequence[Sequence[TimeSeries]],
      Sequence[Sequence[TimeSeries]],
    ],
  ],
) -> None:
  """Save optimization results and confidence intervals to disk."""
