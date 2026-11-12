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
"""Light handling for USD exporter."""

from typing import Any, TypeAlias

import numpy as np
from numpy import typing as npt

_Stage: TypeAlias = Any
_Prim: TypeAlias = Any
_Xform: TypeAlias = Any
_SphereLight: TypeAlias = Any
_DomeLight: TypeAlias = Any
_XformOp: TypeAlias = Any
_Real: TypeAlias = npt.NDArray[np.floating | np.integer]

class USDSphereLight:
  """Class that handles the sphere lights in the USD scene."""

  stage: _Stage
  usd_xform: _Xform
  usd_light: _SphereLight
  usd_prim: _Prim
  translate_op: _XformOp
  def __init__(
    self, stage: _Stage, obj_name: str, radius: float | None = ...
  ) -> None: ...
  def update(
    self, pos: _Real, intensity: int, color: _Real, frame: int
  ) -> None:
    """Updates the attributes of a sphere light."""

class USDDomeLight:
  """Class that handles the dome lights in the USD scene."""

  stage: _Stage
  usd_xform: _Xform
  usd_light: _DomeLight
  usd_prim: _Prim
  def __init__(self, stage: _Stage, obj_name: str) -> None: ...
  def update(self, intensity: int, color: _Real) -> None:
    """Updates the attributes of a dome light."""
