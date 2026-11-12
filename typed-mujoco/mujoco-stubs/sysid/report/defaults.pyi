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

"""Default report generation for system identification results."""

import pathlib
from collections.abc import Callable, Sequence
from typing import TypeAlias

import numpy as np
import scipy.optimize as scipy_optimize
from numpy import typing as npt

from mujoco.sysid._src import parameter
from mujoco.sysid._src.residual import BuildModelFn
from mujoco.sysid._src.timeseries import TimeSeries
from mujoco.sysid._src.trajectory import ModelSequences
from mujoco.sysid.report.builder import ReportBuilder

_ResidualFn: TypeAlias = Callable[
  ...,
  tuple[
    list[npt.NDArray[np.float64]],
    list[list[TimeSeries]],
    list[list[TimeSeries]],
  ],
]

def default_report(
  models_sequences: Sequence[ModelSequences],
  initial_params: parameter.ParameterDict,
  opt_params: parameter.ParameterDict,
  residual_fn: _ResidualFn,
  opt_result: scipy_optimize.OptimizeResult,
  title: str = 'SysID',
  save_path: pathlib.Path | None = None,
  build_model: BuildModelFn | None = ...,
  generate_videos: bool = True,
) -> ReportBuilder: ...
