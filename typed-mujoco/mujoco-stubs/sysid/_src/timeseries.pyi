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

"""Time series utilities."""

import pathlib
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Literal, TypeAlias

import numpy as np
from numpy import typing as npt

import mujoco

class SignalType(Enum):
  MjSensor = 0
  CustomObs = 1
  MjStateQPos = 2
  MjStateQVel = 3
  MjStateAct = 4
  MjCtrl = 5

SignalMappingType: TypeAlias = dict[
  str, tuple[SignalType, npt.NDArray[np.int_]]
]

def encode_signal_mapping(signal_mapping: SignalMappingType) -> str:
  """Serialize a signal mapping to a JSON string."""

def decode_signal_mapping(encoded: str) -> SignalMappingType:
  """Inverse of :func:`encode_signal_mapping`."""

InterpolationMethod: TypeAlias = Literal[
  'linear', 'cubic', 'quadratic', 'quintic', 'zero_order_hold', 'zoh'
]

@dataclass(frozen=True)
class TimeSeries:
  """A utility class for working with time-series data."""

  times: npt.NDArray[np.float64]
  data: npt.NDArray[np.float64]
  signal_mapping: SignalMappingType | None = None

  @staticmethod
  def compute_all_sensor_mapping(model: mujoco.MjModel) -> SignalMappingType:
    """Computes mapping for all sensors in the model."""

  @staticmethod
  def compute_all_control_mapping(model: mujoco.MjModel) -> SignalMappingType:
    """Computes mapping for all controls (actuators) in the model."""

  @staticmethod
  def compute_all_state_mappings(
    model: mujoco.MjModel,
  ) -> tuple[
    SignalMappingType, SignalMappingType, SignalMappingType, SignalMappingType
  ]:
    """Computes mappings for all state components (qpos, qvel, act) + ctrl."""

  @classmethod
  def from_custom_map(
    cls,
    times: npt.NDArray[np.float64],
    data: npt.NDArray[np.float64],
    signals: Sequence[str | tuple[str, int, SignalType]],
  ) -> TimeSeries:
    """Construct a TimeSeries from custom data with explicit signal definitions."""

  @classmethod
  def from_names(
    cls,
    times: npt.NDArray[np.float64],
    data: npt.NDArray[np.float64],
    model: mujoco.MjModel,
    names: Sequence[str | tuple[str, SignalType]] | None = None,
  ) -> TimeSeries:
    """Construct a TimeSeries for observations from the model."""

  @classmethod
  def from_control_names(
    cls,
    times: npt.NDArray[np.float64],
    data: npt.NDArray[np.float64],
    model: mujoco.MjModel,
    names: Sequence[str | tuple[str, SignalType]] | None = None,
  ) -> TimeSeries:
    """Construct a TimeSeries for control signals from the model."""

  def get_indices(
    self, obs_name: str
  ) -> tuple[SignalType, npt.NDArray[np.int_]]:
    """Look up the signal type and column indices for a named observation."""

  @classmethod
  def create(
    cls,
    times: npt.NDArray[np.float64],
    data: npt.NDArray[np.float64],
    signal_mapping: dict[
      str, tuple[SignalType, npt.NDArray[np.int_] | list[int] | int]
    ],
  ) -> TimeSeries:
    """Construct a TimeSeries, normalizing index entries to ``np.ndarray``."""

  @classmethod
  def slice_by_name(
    cls, ts: TimeSeries, enabled_sensors: list[str]
  ) -> TimeSeries:
    """Return a new TimeSeries containing only the named signals."""

  def __post_init__(self) -> None:
    """Validate the time series data after initialization."""

  def __len__(self) -> int: ...
  def save_to_disk(self, path: str | pathlib.Path) -> None:
    """Save the time series data to disk."""

  def save_to_csv(self, path: str | pathlib.Path) -> None:
    """Save the time series data to a CSV file."""

  @classmethod
  def load_from_disk(cls, path: str | pathlib.Path) -> TimeSeries:
    """Load time series data from disk."""

  def interpolate(
    self,
    t: float | npt.NDArray[np.float64],
    method: InterpolationMethod = 'linear',
  ) -> npt.NDArray[np.float64]:
    """Interpolate data at specified time(s)."""

  def get(
    self,
    t: float | npt.NDArray[np.float64],
    method: InterpolationMethod = 'linear',
  ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Get interpolated data at specified time(s)."""

  def resample(
    self,
    new_times: npt.NDArray[np.float64] | None = None,
    target_dt: float | None = None,
    method: InterpolationMethod = 'linear',
  ) -> TimeSeries:
    """Resample the time series to new timestamps or a specific time interval."""

  def remove_from_beginning(self, time_to_remove_s: float) -> TimeSeries:
    """Remove time from the beginning of the time series."""

  def dt_statistics(self) -> dict[str, float]:
    """Calculate statistics about the time intervals."""
