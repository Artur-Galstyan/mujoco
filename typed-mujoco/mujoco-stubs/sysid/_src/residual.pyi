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

"""Residual computation for system identification."""

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, TypeAlias, TypedDict

import numpy as np
from numpy import typing as npt
from typing_extensions import Unpack

import mujoco
from mujoco.sysid._src import parameter, timeseries
from mujoco.sysid._src.trajectory import ModelSequences, SystemTrajectory

BuildModelFn: TypeAlias = Callable[
  [parameter.ParameterDict, mujoco.MjSpec], mujoco.MjModel
]

CustomRolloutFn: TypeAlias = Callable[..., Sequence[SystemTrajectory]]

ModifyResidualFn: TypeAlias = Callable[
  ...,
  tuple[npt.NDArray[np.float64], timeseries.TimeSeries, timeseries.TimeSeries],
]

def construct_ts_from_defaults(
  state_ts: timeseries.TimeSeries,
  pred_sensordata: timeseries.TimeSeries,
  measured_sensordata: timeseries.TimeSeries,
  enabled_observations: Sequence[tuple[str, timeseries.SignalType]]
  | None = None,
) -> tuple[timeseries.TimeSeries, timeseries.TimeSeries]:
  """Assemble predicted observations to match the measured signal layout."""

def model_residual(
  x: npt.NDArray[np.float64],
  params: parameter.ParameterDict,
  build_model: Callable[[parameter.ParameterDict], mujoco.MjModel],
  traj_measured: Sequence[SystemTrajectory] | SystemTrajectory,
  modify_residual: ModifyResidualFn | None = None,
  custom_rollout: CustomRolloutFn | None = None,
  n_threads: int = ...,
  return_pred_all: bool = False,
  resample_true: bool = True,
  sensor_weights: Mapping[str, float] | None = None,
  enabled_observations: Sequence[tuple[str, timeseries.SignalType]] = (),
) -> tuple[
  npt.NDArray[np.float64],
  list[timeseries.TimeSeries],
  list[timeseries.TimeSeries],
]:
  """Compute residuals for a single model against measured trajectories."""

class _ResidualKwargs(TypedDict, total=False):
  models_sequences: list[ModelSequences]
  build_model: BuildModelFn
  modify_residual: ModifyResidualFn | None
  custom_rollout: CustomRolloutFn | None
  n_threads: int
  return_pred_all: bool
  resample_true: bool
  sensor_weights: Mapping[str, float] | None
  enabled_observations: Sequence[tuple[str, timeseries.SignalType]]

class _BuiltResidualFn(Protocol):
  def __call__(
    self,
    x: npt.NDArray[np.float64],
    params: parameter.ParameterDict,
    **kwargs: Unpack[_ResidualKwargs],
  ) -> tuple[
    list[npt.NDArray[np.float64]],
    list[list[timeseries.TimeSeries]],
    list[list[timeseries.TimeSeries]],
  ]: ...

def build_residual_fn(
  **captured_kwargs: Unpack[_ResidualKwargs],
) -> _BuiltResidualFn:
  """Create a residual closure with pre-bound keyword arguments."""

def residual(
  x: npt.NDArray[np.float64],
  params: parameter.ParameterDict,
  models_sequences: list[ModelSequences],
  build_model: BuildModelFn = ...,
  modify_residual: ModifyResidualFn | None = None,
  custom_rollout: CustomRolloutFn | None = None,
  n_threads: int = ...,
  return_pred_all: bool = False,
  resample_true: bool = True,
  sensor_weights: Mapping[str, float] | None = None,
  enabled_observations: Sequence[tuple[str, timeseries.SignalType]] = (),
) -> tuple[
  list[npt.NDArray[np.float64]],
  list[list[timeseries.TimeSeries]],
  list[list[timeseries.TimeSeries]],
]:
  """Top-level residual: iterate over all model-sequence groups."""
