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

"""Common signal modifiers."""

from collections.abc import Mapping

import numpy as np
from numpy import typing as npt

import mujoco
from mujoco.sysid._src import parameter, timeseries

def get_sensor_indices(
  model: mujoco.MjModel,
  sensor_name: str | list[str],
  sort: bool = False,
) -> list[int]:
  """Get sensor indices from a sensor configuration dictionary."""

def apply_bias(
  ts: timeseries.TimeSeries,
  sensor_name: str,
  bias: parameter.Parameter,
) -> timeseries.TimeSeries:
  """Apply a bias to a sensor in a timeseries."""

def apply_gain(
  ts: timeseries.TimeSeries,
  sensor_name: str,
  gain: parameter.Parameter,
) -> timeseries.TimeSeries:
  """Apply a gain to a sensor in a timeseries."""

def apply_delay(
  ts: timeseries.TimeSeries,
  sensor_name: str,
  delay: parameter.Parameter,
) -> timeseries.TimeSeries:
  """Apply a delay to a sensor in a timeseries."""

def apply_time_window(
  ts: timeseries.TimeSeries,
  min_t: float,
  max_t: float,
) -> timeseries.TimeSeries:
  """Select a subset of a timeseries within a time window."""

def apply_delayed_ts_window(
  ts: timeseries.TimeSeries,
  ts_delayed: timeseries.TimeSeries,
  min_delay: float,
  max_delay: float,
) -> timeseries.TimeSeries:
  """Window a timeseries to fit within the bounds of a delayed series."""

def apply_resample_and_delay(
  ts: timeseries.TimeSeries,
  times: npt.NDArray[np.float64],
  default_delay: float,
  sensor_delays: dict[str, float] | None = None,
  predicted_data: bool = True,
) -> timeseries.TimeSeries:
  """Resample a timeseries and apply per-sensor delays."""

def prepare_sensor_weights(
  sensor_weights: Mapping[str, float] | npt.NDArray[np.float64],
  n_sensors: int,
  model: mujoco.MjModel,
) -> npt.NDArray[np.float64]:
  """Prepare sensor weights array from a dict or numpy array."""

def weighted_diff(
  predicted_data: npt.NDArray[np.float64],
  measured_data: npt.NDArray[np.float64],
  model: mujoco.MjModel | None = None,
  sensor_weights: Mapping[str, float] | npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
  """Compute the weighted difference between measured and predicted data."""

def normalize_residual(
  residual: npt.NDArray[np.float64],
  measured_data: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
  """Normalize the residual by the standard deviation of the measured data."""
