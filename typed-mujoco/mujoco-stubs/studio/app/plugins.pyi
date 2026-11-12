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
"""Python implementation of the default Studio Viewer application UI."""

import dataclasses

import mujoco
from mujoco.studio import messages, plugin, sim, ux

SIM_PLUGINS: tuple[plugin.SimPlugin, ...]

def install_console_hooks() -> None:
  """Add logs, warnings, and uncaught exceptions to the Studio Console."""

@dataclasses.dataclass(frozen=True)
class InitEvent(messages.Event):
  """Lifecycle event dispatched once when the studio_app App starts."""

  app: App

@dataclasses.dataclass
class Config:
  """Configure the studio_app App."""

  show_info: bool = False
  show_profiler: bool = False
  show_console: bool = False
  show_options: bool = True
  show_toolbar: bool = False
  show_view_gizmo: bool = True
  show_status_bar: bool = True
  show_inspector: bool = True

class App(plugin.ViewerPlugin):
  """Wrap a Viewer and add Studio UI/UX."""

  theme: ux.GuiTheme
  config: Config
  status: str
  step_control_state: sim.StepControl
  ux_state: ux.UxState
  @property
  def model(self) -> mujoco.MjModel: ...
  @model.setter
  def model(self, value: mujoco.MjModel) -> None: ...
  @property
  def model_path(self) -> str: ...
  @model_path.setter
  def model_path(self, value: str) -> None: ...
  @property
  def data(self) -> mujoco.MjData: ...
  @data.setter
  def data(self, value: mujoco.MjData) -> None: ...
  def __init__(self, config: Config | None = None) -> None: ...
  def align_camera(self) -> None:
    """Recenter the camera on the home camera, else the free camera."""
  def reload_model(self) -> None:
    """Ask the sim to reload the current model."""
  def close(self) -> None: ...
  def handle_keyboard_events(self) -> None:
    """Handle keyboard events."""
  def handle_camera_tracking_mouse_events(self) -> None:
    """Handle mouse events for camera tracking."""
  def handle_mouse_events(self) -> None:
    """Handle mouse events."""
  def reset_physics(self) -> None:
    """Reset the physics."""
  def save_keyframe(self) -> None:
    """Save the active state to the selected keyframe and notify the sim."""
  def apply_perturb(self) -> None:
    """Apply the perturbation to the model."""
  def reset_physics_gui(self) -> None:
    """Draw the reset button."""
  def is_running(self) -> bool:
    """Return True if the application should continue running."""
  def update(self) -> None:
    """Update the simulation and handle user input."""
  def build_gui(self) -> None:
    """Emit the full Studio UI."""
