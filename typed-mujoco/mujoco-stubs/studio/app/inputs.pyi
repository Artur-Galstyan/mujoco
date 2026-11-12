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
"""Keyboard and mouse input handling for the Studio app."""

import mujoco
from mujoco.studio import sim, ux
from mujoco.studio.app import plugins

def handle_vis_options_keyboard_events(
  vis_options: mujoco.MjvOption, is_freecam_wasd: bool
) -> bool:
  """Toggle visualization flags based on keyboard shortcuts."""

def handle_step_control_keyboard_events(
  step_control: sim.StepControl, ux_state: ux.UxState
) -> bool:
  """Handle keyboard shortcuts for simulation stepping control."""

def handle_sim_history_keyboard_events(
  step_control: sim.StepControl, ux_state: ux.UxState
) -> bool:
  """Handle keyboard shortcuts for scrubbing the simulation history."""

def handle_reset_keyboard_events(
  model: mujoco.MjModel, data: mujoco.MjData
) -> bool:
  """Handle keyboard shortcuts for simulation reset."""

def handle_camera_select_keyboard_events(
  model: mujoco.MjModel, camera: mujoco.MjvCamera, ux_state: ux.UxState
) -> bool:
  """Handle keyboard shortcuts for camera selection."""

def handle_miscellaneous_keyboard_events(app: plugins.App) -> bool:
  """Handle keyboard shortcuts for toggling panels and windows."""

def handle_freecam_wasd_keyboard_events(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  camera: mujoco.MjvCamera,
  cam_speed: float,
) -> tuple[bool, float]:
  """Handle keyboard shortcuts for free camera movement."""

def handle_keyboard_events(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  camera: mujoco.MjvCamera,
  vis_options: mujoco.MjvOption,
  step_control: sim.StepControl,
  ux_state: ux.UxState,
  cam_speed: float,
) -> tuple[bool, float]:
  """Handle keyboard events according to the Studio bindings."""

def handle_camera_tracking_mouse_events(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  camera: mujoco.MjvCamera,
  vis_options: mujoco.MjvOption,
  ux_state: ux.UxState,
) -> None:
  """Handle mouse events for camera tracking."""

def handle_mouse_events(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  camera: mujoco.MjvCamera,
  vis_options: mujoco.MjvOption,
  perturb: mujoco.MjvPerturb,
  ux_state: ux.UxState,
  move_camera: bool = True,
) -> None:
  """Handle mouse events."""
