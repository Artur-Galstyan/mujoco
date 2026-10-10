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
"""Class for coordinating the rendering of scenes to render targets."""

from typing import Any, Protocol, TypeAlias

from typing_extensions import Buffer, CapsuleType

from mujoco import MjvGLCamera, mjrf

class SupportsDLPack(Protocol):
  """Protocol for objects supporting the DLPack tensor exchange protocol."""

  # A stream can be an opaque, device-specific GPU handle.
  def __dlpack__(self, /, *, stream: Any = None) -> CapsuleType: ...
  def __dlpack_device__(self, /) -> tuple[int, int]: ...

BufferType: TypeAlias = Buffer | SupportsDLPack

class Renderer:
  """Coordinates the rendering of scenes to render targets.

  Internally, the Renderer manages the mjrf objects (i.e. Scene, Camera, and
  RenderTarget) that are needed to render a MuJoCo model. The Scene defines
  the entities from the model that are to be rendered. The Camera defines the
  viewpoint from which to render the model. And, finally, the RenderTarget
  defines the memory buffers into which the final rendered image will be
  written.

  Each object owned by the Renderer is associated with a unique name. This name
  is used to identify the object when making rendering calls.

  The Renderer does not specify the contents of the Scene; it is up to the user
  to populate the Scene with the entities that they want to render.

  The Renderer is not limited to rendering a single scene to a single target.
  You can configure the Renderer to manage multiple scenes from multiple
  cameras into multiple targets. Each such render operation is specified using
  the view() method.
  """

  def __init__(self, ctx: mjrf.Context) -> None: ...
  def scene(self, name: str) -> mjrf.Scene:
    """Returns the scene with the given name, creating it if needed."""
  def target(
    self,
    name: str,
    buffer: BufferType,
    pixel_format: mjrf.PixelFormat = ...,
  ) -> mjrf.RenderTarget:
    """Defines a render target with the given name and destination buffer."""
  def view(
    self,
    name: str,
    *,
    scene: str,
    target: str,
    camera: mjrf.Camera | MjvGLCamera = ...,
    viewport: tuple[float, float, float, float] | None = None,
    draw_mode: mjrf.DrawMode = ...,
  ) -> None:
    """Associates a scene, camera, viewport, and target as a rendering operation."""
  def update_camera(self, name: str, camera: mjrf.Camera | MjvGLCamera) -> None:
    """Updates the camera for the given view."""
  def render(self) -> None:
    """Renders all active views."""
