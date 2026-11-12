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
"""Camera handling for USD exporter."""

from typing import Any, TypeAlias

import numpy as np
from numpy import typing as npt

_Stage: TypeAlias = Any
_Prim: TypeAlias = Any
_Xform: TypeAlias = Any
_Camera: TypeAlias = Any
_XformOp: TypeAlias = Any
_Real: TypeAlias = npt.NDArray[np.floating | np.integer]

class USDCamera:
  """Class that handles the cameras in the USD scene."""

  stage: _Stage
  usd_xform: _Xform
  usd_camera: _Camera
  usd_prim: _Prim
  transform_op: _XformOp
  def __init__(self, stage: _Stage, obj_name: str) -> None: ...
  def update(self, cam_pos: _Real, cam_mat: _Real, frame: int) -> None:
    """Updates the position and orientation of the camera in the scene."""
