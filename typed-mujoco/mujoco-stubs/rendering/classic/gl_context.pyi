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

"""Select an OpenGL context backend according to `MUJOCO_GL` and the OS.

GLFW requires a display. Linux also supports headless EGL or OSMesa.
Set `MUJOCO_GL` before importing MuJoCo to choose a backend.
"""

class GLContext:
  """Own a classic OpenGL context for offscreen MuJoCo rendering.

  Call `make_current` before rendering and `free` when finished. For most
  users, `Renderer` creates and frees the context automatically.

  Example (requires a working OpenGL backend; GLFW needs a GUI display):
    ```python
    import mujoco
    from mujoco.rendering.classic.gl_context import GLContext

    model = mujoco.MjModel.from_xml_string('<mujoco/>')
    context = GLContext(320, 240)
    try:
      context.make_current()
      mjr = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150.value)
      try:
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN.value, mjr)
      finally:
        mjr.free()
    finally:
      context.free()
    ```
  """

  def __init__(self, max_width: int, max_height: int) -> None:
    """Create a context with maximum width and height for buffer-backed GL."""
  def make_current(self) -> None:
    """Make this context current before issuing rendering commands."""
  def free(self) -> None:
    """Release the OpenGL context and its resources."""
