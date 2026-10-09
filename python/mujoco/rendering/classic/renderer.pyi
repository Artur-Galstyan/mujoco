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

from types import TracebackType
from typing import TypeVar, overload

import numpy as np
from numpy import typing as npt
from typing_extensions import Self

from mujoco import MjData, MjModel, MjvCamera, MjvOption, MjvScene, mjtFontScale
from mujoco.rendering.classic import gl_context

_ScalarT = TypeVar('_ScalarT', bound=np.generic)
GLContext: type[gl_context.GLContext] | None

class Renderer:
  def __init__(
    self,
    model: MjModel,
    height: int = 240,
    width: int = 320,
    max_geom: int = 10000,
    font_scale: mjtFontScale = ...,
  ) -> None: ...
  @property
  def model(self) -> MjModel: ...
  @property
  def scene(self) -> MjvScene: ...
  @property
  def height(self) -> int: ...
  @property
  def width(self) -> int: ...
  def enable_depth_rendering(self) -> None: ...
  def disable_depth_rendering(self) -> None: ...
  def enable_segmentation_rendering(self) -> None: ...
  def disable_segmentation_rendering(self) -> None: ...
  @overload
  def render(
    self, *, out: None = None
  ) -> (
    npt.NDArray[np.uint8] | npt.NDArray[np.float32] | npt.NDArray[np.int32]
  ): ...
  @overload
  def render(
    self, *, out: npt.NDArray[_ScalarT]
  ) -> npt.NDArray[_ScalarT] | npt.NDArray[np.int32]: ...
  def update_scene(
    self,
    data: MjData,
    camera: int | str | MjvCamera = -1,
    scene_option: MjvOption | None = None,
  ) -> None: ...
  def close(self) -> None: ...
  def __enter__(self) -> Self: ...
  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc_value: BaseException | None,
    traceback: TracebackType | None,
  ) -> None: ...
