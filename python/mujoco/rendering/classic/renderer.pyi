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

"""Render MuJoCo scenes to NumPy arrays with a classic OpenGL backend."""

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
  """Render a scene into RGB, depth, or object segmentation images.

  Requires an available OpenGL backend; GLFW also needs a display.
  Use a context manager to release the graphics contexts after use.

  Example (requires OpenGL, not a GUI-free doctest):
    ```python
    import mujoco

    model = mujoco.MjModel.from_xml_string(
      '<mujoco><worldbody><geom size=".1"/></worldbody></mujoco>'
    )
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with mujoco.Renderer(model, height=240, width=320) as renderer:
      renderer.update_scene(data)
      rgb = renderer.render()
      assert rgb.shape == (240, 320, 3)
    ```
  """

  def __init__(
    self,
    model: MjModel,
    height: int = 240,
    width: int = 320,
    max_geom: int = 10000,
    font_scale: mjtFontScale = ...,
  ) -> None:
    """Create offscreen contexts for an image of `height` by `width` pixels.

    The size must fit the model's offscreen framebuffer. `max_geom` limits
    the number of scene geoms; `font_scale` sets the rendering font size.
    """
  @property
  def model(self) -> MjModel:
    """The model used to create this renderer."""
  @property
  def scene(self) -> MjvScene:
    """The scene last built by `update_scene`, available for customization."""
  @property
  def height(self) -> int:
    """Image height in pixels."""
  @property
  def width(self) -> int:
    """Image width in pixels."""
  def enable_depth_rendering(self) -> None:
    """Select metric depth images and turn off segmentation rendering."""
  def disable_depth_rendering(self) -> None:
    """Return to RGB images after depth rendering."""
  def enable_segmentation_rendering(self) -> None:
    """Select object ID/type images and turn off depth rendering."""
  def disable_segmentation_rendering(self) -> None:
    """Return to RGB images after segmentation rendering."""
  @overload
  def render(
    self, *, out: None = None
  ) -> npt.NDArray[np.uint8] | npt.NDArray[np.float32] | npt.NDArray[np.int32]:
    """Render the scene last built by `update_scene`.

    RGB returns `(height, width, 3)` uint8. Depth returns `(height, width)`
    float32 in distance units. Segmentation returns `(height, width, 2)`
    int32: channel 0 is object ID and channel 1 is object type. Background
    pixels are `(-1, -1)`. Call the matching enable/disable method to change
    modes; RGB is the default. Rendering after `close` raises RuntimeError.
    """
  @overload
  def render(
    self, *, out: npt.NDArray[_ScalarT]
  ) -> npt.NDArray[_ScalarT] | npt.NDArray[np.int32]:
    """Write RGB or depth pixels to `out` and return it.

    Use a uint8 buffer of shape `(height, width, 3)` for RGB or segmentation,
    or float32 of shape `(height, width)` for depth. Segmentation uses `out`
    as a temporary RGB buffer and returns a separate int32 array of shape
    `(height, width, 2)`.
    """
  def update_scene(
    self,
    data: MjData,
    camera: int | str | MjvCamera = -1,
    scene_option: MjvOption | None = None,
  ) -> None:
    """Rebuild the scene from `data` before rendering.

    `camera=-1` uses a free camera; pass a fixed camera ID, name, or an
    `MjvCamera` for another view. `scene_option` overrides the default options.
    An unknown camera name or invalid ID raises ValueError.
    """
  def close(self) -> None:
    """Free the OpenGL and MuJoCo render contexts; `render` then fails."""
  def __enter__(self) -> Self: ...
  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc_value: BaseException | None,
    traceback: TracebackType | None,
  ) -> None: ...
