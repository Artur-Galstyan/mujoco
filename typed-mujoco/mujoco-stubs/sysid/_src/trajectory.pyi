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

"""Trajectory data containers for system identification."""

import pathlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypedDict

import numpy as np
from numpy import typing as npt
from typing_extensions import Unpack

import mujoco
from mujoco.sysid._src import timeseries

class _TrajectoryFields(TypedDict, total=False):
  model: mujoco.MjModel
  control: timeseries.TimeSeries
  sensordata: timeseries.TimeSeries
  initial_state: npt.NDArray[np.float64]
  state: timeseries.TimeSeries | None

@dataclass(frozen=True)
class SystemTrajectory:
  """Encapsulates a trajectory rolled out from a system."""

  model: mujoco.MjModel
  control: timeseries.TimeSeries
  sensordata: timeseries.TimeSeries
  initial_state: npt.NDArray[np.float64]
  state: timeseries.TimeSeries | None

  def replace(self, **kwargs: Unpack[_TrajectoryFields]) -> SystemTrajectory:
    """Return a copy with the specified fields replaced."""

  def get_sensordata_slice(
    self, sensor: str = 'joint_pos'
  ) -> npt.NDArray[np.float64]:
    """Extract sensor columns by type."""

  @property
  def sensordim(self) -> int:
    """Total number of scalar sensor outputs in the model."""

  def __len__(self) -> int:
    """Number of time steps in the trajectory."""

  def save_to_disk(self, path: pathlib.Path) -> None:
    """Save this trajectory to disk as a compressed NumPy archive."""

  @classmethod
  def load_from_disk(
    cls,
    path: pathlib.Path,
    model: mujoco.MjModel,
    allow_missing_sensors: bool = False,
  ) -> SystemTrajectory:
    """Load a trajectory from a compressed NumPy archive."""

  def check_compatible(self, allow_missing_sensors: bool = False) -> None:
    """Validate that data dimensions match the model."""

  def split(self, chunk_size: int) -> list[SystemTrajectory]:
    """Split into consecutive non-overlapping chunks of *chunk_size* steps."""

  def render(
    self,
    height: int = 240,
    width: int = 320,
    camera: str | int = -1,
    fps: int = 30,
  ) -> list[npt.NDArray[np.uint8]]:
    """Render this trajectory to a list of RGB frames."""

def create_initial_state(
  model: mujoco.MjModel,
  qpos: npt.NDArray[np.float64],
  qvel: npt.NDArray[np.float64] | None = None,
  act: npt.NDArray[np.float64] | None = None,
  qpos_names: Sequence[str] | None = None,
  qvel_names: Sequence[str] | None = None,
  act_names: Sequence[str] | None = None,
) -> npt.NDArray[np.float64]:
  """Build a ``mjSTATE_FULLPHYSICS`` initial-state vector from components."""

class ModelSequences:
  """A model spec paired with one or more measured trajectory sequences."""

  name: str
  spec: mujoco.MjSpec
  allow_missing_sensors: bool
  gt_model: mujoco.MjModel
  sequence_name: list[str]
  initial_state: list[npt.NDArray[np.float64]]
  control: list[timeseries.TimeSeries]
  sensordata: list[timeseries.TimeSeries]
  measured_rollout: list[SystemTrajectory]

  def __init__(
    self,
    name: str,
    spec: mujoco.MjSpec,
    sequence_name: str | Sequence[str],
    initial_state: npt.NDArray[np.float64] | Sequence[npt.NDArray[np.float64]],
    control: timeseries.TimeSeries | Sequence[timeseries.TimeSeries],
    sensordata: timeseries.TimeSeries | Sequence[timeseries.TimeSeries],
    allow_missing_sensors: bool = False,
  ) -> None: ...
  def __getitem__(self, key: int | slice) -> ModelSequences: ...

def timeseries2array(
  control_signal: timeseries.TimeSeries | Sequence[timeseries.TimeSeries],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
  """Convert control TimeSeries to stacked arrays, dropping the last step."""

def sequence2array(
  initial_states: npt.NDArray[np.float64] | Sequence[npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
  """Stack a sequence of initial-state vectors into a single array."""

def arrays2traj(
  models: mujoco.MjModel | Sequence[mujoco.MjModel],
  initial_states: npt.NDArray[np.float64] | Sequence[npt.NDArray[np.float64]],
  control: npt.NDArray[np.float64],
  control_times: npt.NDArray[np.float64],
  state: npt.NDArray[np.float64],
  sensordata: npt.NDArray[np.float64],
  signal_mapping: timeseries.SignalMappingType,
  state_mapping: timeseries.SignalMappingType,
  ctrl_mapping: timeseries.SignalMappingType,
) -> Sequence[SystemTrajectory]:
  """Convert raw rollout arrays into a list of SystemTrajectory objects."""

def sysid_rollout(
  models: mujoco.MjModel | Sequence[mujoco.MjModel],
  datas: mujoco.MjData | Sequence[mujoco.MjData],
  control_signal: Sequence[timeseries.TimeSeries] | timeseries.TimeSeries,
  initial_states: npt.NDArray[np.float64] | Sequence[npt.NDArray[np.float64]],
  rollout_signal_mapping: timeseries.SignalMappingType | None = None,
  rollout_state_mapping: timeseries.SignalMappingType | None = None,
  ctrl_mapping: timeseries.SignalMappingType | None = None,
) -> Sequence[SystemTrajectory]:
  """Rollout trajectories in parallel for the given models and controls."""
