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
"""USD exporter."""

from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy import typing as npt

import mujoco
from mujoco.usd.camera import USDCamera
from mujoco.usd.lights import USDSphereLight
from mujoco.usd.objects import USDObject

_Stage: TypeAlias = Any
_Real: TypeAlias = npt.NDArray[np.floating | np.integer]

class USDExporter:
  """MuJoCo to USD exporter for porting scenes to external renderers."""

  model: mujoco.MjModel
  height: int
  width: int
  max_geom: int
  output_directory: str
  output_directory_root: str
  light_intensity: int
  camera_names: list[str] | None
  specialized_materials_file: str | None
  verbose: bool
  frame_count: int
  updates: int
  geom_names: set[str]
  geom_refs: dict[str, USDObject]
  usd_lights: list[USDSphereLight | None]
  usd_cameras: list[USDCamera]
  stage: _Stage
  output_directory_path: str
  frames_directory: str
  assets_directory: str
  texture_files: list[str]
  def __init__(
    self,
    model: mujoco.MjModel,
    height: int = ...,
    width: int = ...,
    max_geom: int = ...,
    output_directory: str = ...,
    output_directory_root: str = ...,
    light_intensity: int = ...,
    camera_names: list[str] | None = ...,
    specialized_materials_file: str | None = ...,
    verbose: bool = ...,
  ) -> None:
    """Initializes a new USD Exporter."""

  @property
  def usd(self) -> str:
    """Returns the USD file as a string."""

  @property
  def scene(self) -> mujoco.MjvScene:
    """Returns the scene."""

  def update_scene(
    self,
    data: mujoco.MjData,
    scene_option: mujoco.MjvOption | None = ...,
  ) -> None:
    """Updates the scene with latest sim data."""

  def add_light(
    self,
    pos: Sequence[float],
    intensity: int,
    radius: float | None = ...,
    color: _Real | None = ...,
    obj_name: str | None = ...,
    light_type: Literal['sphere', 'dome'] | None = ...,
  ) -> None:
    """Adds a user defined, fixed light."""

  def add_camera(
    self,
    pos: Sequence[float],
    rotation_xyz: Sequence[float],
    obj_name: str | None = ...,
  ) -> None:
    """Adds a user defined, fixed camera."""

  def save_scene(self, filetype: Literal['usd', 'usda', 'usdc'] = ...) -> None:
    """Saves the scene to a USD file."""
