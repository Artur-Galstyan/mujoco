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

"""Declarative signal transformation for system identification residuals."""

from collections.abc import Mapping

import numpy as np
from numpy import typing as npt

import mujoco
from mujoco.sysid._src import parameter, timeseries

class SignalTransform:
  """Declarative signal transformation replacing modify_residual callbacks."""

  normalize: bool

  def __init__(self, normalize: bool = True) -> None: ...
  def delay(self, pattern: str, param: parameter.Parameter) -> None:
    """Register a delay for sensors matching *pattern* (fnmatch)."""

  def gain(
    self,
    pattern: str,
    param: parameter.Parameter,
    target: str = 'both',
  ) -> None:
    """Register a multiplicative gain for sensors matching *pattern*."""

  def bias(
    self,
    pattern: str,
    param: parameter.Parameter,
    target: str = 'both',
  ) -> None:
    """Register an additive bias for sensors matching *pattern*."""

  def enable_sensors(self, sensor_names: list[str]) -> None:
    """Only include these sensors in the returned residual/timeseries."""

  def set_sensor_weights(self, weights: Mapping[str, float]) -> None:
    """Set per-sensor weights for the weighted diff."""

  def apply(
    self,
    params: parameter.ParameterDict,
    sensordata_predicted: timeseries.TimeSeries,
    sensordata_measured: timeseries.TimeSeries,
    model: mujoco.MjModel,
    return_pred_all: bool,
    state: npt.NDArray[np.float64] | None = None,
    sensor_weights: Mapping[str, float] | None = None,
  ) -> tuple[
    npt.NDArray[np.float64], timeseries.TimeSeries, timeseries.TimeSeries
  ]:
    """Apply all registered transforms and compute the residual."""
