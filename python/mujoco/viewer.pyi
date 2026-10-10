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

"""Open a MuJoCo viewer with a GUI and a working OpenGL display.

On macOS, run scripts that call `launch_passive` with `mjpython`.
"""

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
  """Control a passive viewer while your code advances the simulation.

  The viewer does not step the simulation. Use `lock` while changing the
  shared model or data, then call `sync` outside the lock. Closing the window
  stops `is_running`; a context manager closes the viewer on exit.

  Example (requires a GUI/OpenGL display; on macOS use `mjpython script.py`):
    ```python
    import time
    import mujoco
    from mujoco import viewer

    model = mujoco.MjModel.from_xml_string(
      '<mujoco><worldbody><geom size=".1"/></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    with viewer.launch_passive(model, data) as handle:
      while handle.is_running():
        with handle.lock():
          mujoco.mj_step(model, data)
        handle.sync()
        time.sleep(model.opt.timestep)
    ```
  """

  def __init__(
    self,
    sim: _Simulate,
    cam: MjvCamera,
    opt: MjvOption,
    pert: MjvPerturb,
    user_scn: MjvScene | None,
  ) -> None: ...
  @property
  def cam(self) -> MjvCamera:
    """The viewer's mutable camera; protect edits with `lock`."""
  @property
  def opt(self) -> MjvOption:
    """The viewer's mutable visualization options."""
  @property
  def perturb(self) -> MjvPerturb:
    """The viewer's mutable mouse-perturbation state."""
  @property
  def user_scn(self) -> MjvScene | None:
    """The passive viewer's extra geoms and flags, copied at `sync`."""
  @property
  def m(self) -> MjModel | None:
    """The shared model, or None after the viewer is destroyed."""
  @property
  def d(self) -> MjData | None:
    """The shared data, or None after the viewer is destroyed."""
  @property
  def viewport(self) -> MjrRect | None:
    """The current pixel viewport, or None after viewer destruction."""
  def set_figures(
    self,
    viewports_figures: tuple[MjrRect, MjvFigure]
    | list[tuple[MjrRect, MjvFigure]],
  ) -> None:
    """Display one `(viewport, figure)` pair or a list of such pairs."""
  def clear_figures(self) -> None:
    """Remove all figure overlays."""
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
  ) -> None:
    """Display one or more `(font, gridpos, left, right)` text overlays.

    None uses the default font or top-left position, or an empty text column.
    """
  def clear_texts(self) -> None:
    """Remove all text overlays."""
  def set_images(
    self,
    viewports_images: tuple[MjrRect, npt.NDArray[np.uint8]]
    | list[tuple[MjrRect, npt.NDArray[np.uint8]]],
  ) -> None:
    """Display one or more `(viewport, image)` overlays.

    Each image is an RGB uint8 array with shape
    `(viewport.height, viewport.width, 3)`. The viewer copies and flips it.
    """
  def clear_images(self) -> None:
    """Remove all image overlays."""
  def close(self) -> None:
    """Request that the viewer exit; also called on context manager exit."""
  def is_running(self) -> bool:
    """Return False after the viewer closes or an exit is requested."""
  def lock(self) -> AbstractContextManager[None]:
    """Protect edits to the shared model, data, or viewer state.

    Do not call `sync` inside this lock: `sync` acquires the lock itself.
    """
  def sync(self, state_only: bool = False) -> None:
    """Send simulation changes to the viewer and apply GUI changes.

    `state_only=True` copies integration state instead of the full model and
    data to the viewer. Call outside `lock`; this method locks internally.
    """
  def update_hfield(self, hfieldid: int) -> None:
    """Upload changes to the model's height field at `hfieldid`."""
  def update_mesh(self, meshid: int) -> None:
    """Upload changes to the model's mesh at `meshid`."""
  def update_texture(self, texid: int) -> None:
    """Upload changes to the model's texture at `texid`."""
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
) -> None:
  """Run the interactive viewer and its physics loop until it closes.

  Pass a model and optional data, or a `loader` that returns both for reloads.
  The two UI flags control the left and right side panels. Requires a GUI.
  """

def launch_from_path(path: str) -> None:
  """Open an MJCF/XML or MJB model file in the blocking GUI viewer."""

def launch_passive(
  model: MjModel,
  data: MjData,
  *,
  key_callback: KeyCallbackType | None = None,
  show_left_ui: bool = True,
  show_right_ui: bool = True,
) -> Handle:
  """Open a passive GUI viewer and return its `Handle` without blocking.

  Your code owns stepping `data`; call `Handle.sync` to show each new state.
  `key_callback` receives a GLFW key code; the UI flags control side panels.
  Requires a working GUI/OpenGL display. On macOS, use `mjpython` to run the
  script so viewer UI calls occur on the main thread (see `Handle` example).
  """
