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

from collections.abc import Callable
from contextlib import AbstractContextManager
from types import TracebackType
from typing import TypeAlias

import numpy as np
from numpy import typing as npt
from typing_extensions import Self

from mujoco import (
  MjData,
  MjModel,
  MjrRect,
  MjvCamera,
  MjvFigure,
  MjvOption,
  MjvPerturb,
  MjvScene,
  mjtFontScale,
  mjtGridPos,
)
from mujoco._simulate import Simulate as _Simulate

PERCENT_REALTIME: tuple[float, ...]
MAX_SYNC_MISALIGN: float
SIM_REFRESH_FRACTION: float

CallbackType: TypeAlias = Callable[[MjModel, MjData], None]
LoaderType: TypeAlias = Callable[[], tuple[MjModel, MjData]]
KeyCallbackType: TypeAlias = Callable[[int], None]

class Handle:
  def __init__(
    self,
    sim: _Simulate,
    cam: MjvCamera,
    opt: MjvOption,
    pert: MjvPerturb,
    user_scn: MjvScene | None,
  ) -> None: ...
  @property
  def cam(self) -> MjvCamera: ...
  @property
  def opt(self) -> MjvOption: ...
  @property
  def perturb(self) -> MjvPerturb: ...
  @property
  def user_scn(self) -> MjvScene | None: ...
  @property
  def m(self) -> MjModel | None: ...
  @property
  def d(self) -> MjData | None: ...
  @property
  def viewport(self) -> MjrRect | None: ...
  def set_figures(
    self,
    viewports_figures: tuple[MjrRect, MjvFigure]
    | list[tuple[MjrRect, MjvFigure]],
  ) -> None: ...
  def clear_figures(self) -> None: ...
  def set_texts(
    self,
    texts: tuple[
      int | mjtFontScale | None,
      int | mjtGridPos | None,
      str | None,
      str | None,
    ]
    | list[
      tuple[
        int | mjtFontScale | None,
        int | mjtGridPos | None,
        str | None,
        str | None,
      ]
    ],
  ) -> None: ...
  def clear_texts(self) -> None: ...
  def set_images(
    self,
    viewports_images: tuple[MjrRect, npt.NDArray[np.uint8]]
    | list[tuple[MjrRect, npt.NDArray[np.uint8]]],
  ) -> None: ...
  def clear_images(self) -> None: ...
  def close(self) -> None: ...
  def is_running(self) -> bool: ...
  def lock(self) -> AbstractContextManager[None]: ...
  def sync(self, state_only: bool = False) -> None: ...
  def update_hfield(self, hfieldid: int) -> None: ...
  def update_mesh(self, meshid: int) -> None: ...
  def update_texture(self, texid: int) -> None: ...
  def __enter__(self) -> Self: ...
  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc_val: BaseException | None,
    exc_tb: TracebackType | None,
  ) -> None: ...

def launch(
  model: MjModel | None = None,
  data: MjData | None = None,
  *,
  loader: LoaderType | None = None,
  show_left_ui: bool = True,
  show_right_ui: bool = True,
) -> None: ...
def launch_from_path(path: str) -> None: ...
def launch_passive(
  model: MjModel,
  data: MjData,
  *,
  key_callback: KeyCallbackType | None = None,
  show_left_ui: bool = True,
  show_right_ui: bool = True,
) -> Handle: ...
