# Copyright 2026 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS.
# Modified for typed-mujoco 3.15.0 compatibility.

from collections.abc import Callable, Sequence
from typing import ClassVar

import numpy as np
from numpy import typing as npt

from mujoco import MjData, MjModel, MjSpec, MjvCamera, MjvOption, MjvPerturb
from mujoco.experimental.studio.sim import SimHistory, StepControl

class RenderFlags:
  """Studio render flags. Reading flags returns a copy; assign to write them."""

  def __init__(self) -> None: ...
  @property
  def flags(self) -> list[int]: ...
  @flags.setter
  def flags(self, value: Sequence[int] | npt.NDArray[np.uint8]) -> None: ...

class GuiTheme:
  __members__: ClassVar[dict[str, GuiTheme]]
  LIGHT: ClassVar[GuiTheme]
  DARK: ClassVar[GuiTheme]
  CLASSIC: ClassVar[GuiTheme]
  def __init__(self, value: int) -> None: ...
  def __int__(self) -> int: ...
  def __index__(self) -> int: ...
  @property
  def name(self) -> str: ...
  @property
  def value(self) -> int: ...

class UxState:
  """Mutable UI state shared between Studio widgets and the simulation."""

  def __init__(self) -> None: ...
  speed_index: int
  state_sig: int
  watch_field_index: int
  camera_index: int
  key_idx: int
  nthread: int
  update_threadpool: bool
  @property
  def state(self) -> list[float]: ...
  @state.setter
  def state(
    self,
    value: Sequence[float] | npt.NDArray[np.float32] | npt.NDArray[np.float64],
  ) -> None: ...
  sim_head_time: float
  history_size: int
  history_index: int
  watch_field_name: str

def setup_theme(theme: GuiTheme) -> None: ...
def set_imgui_context(ptr: int) -> None: ...
def set_implot_context(ptr: int) -> None: ...
def configure_docking_layout(
  show_toolbar: bool = True, show_status_bar: bool = True
) -> tuple[float, float, float, float]: ...
def step_control_gui(step_control: StepControl, ux_state: UxState) -> None: ...
def reset_history(
  history: SimHistory, ux_state: UxState, model: MjModel, data: MjData
) -> None: ...
def record_history(
  history: SimHistory, ux_state: UxState, model: MjModel, data: MjData
) -> None: ...
def load_history_frame(
  history: SimHistory, model: MjModel, data: MjData, index: int
) -> None: ...
def simulation_gui(
  model: MjModel,
  data: MjData,
  step_control: StepControl,
  ux_state: UxState,
  reset: Callable[[], object],
  reload: Callable[[], object],
  align: Callable[[], object],
) -> None: ...
def timeline_scrubber_gui(
  model: MjModel, data: MjData, step_control: StepControl, ux_state: UxState
) -> None: ...
def theme_select_gui(theme: GuiTheme) -> tuple[bool, GuiTheme]: ...
def theme_menu_gui(theme: GuiTheme) -> tuple[bool, GuiTheme]: ...
def get_default_ini_path() -> str: ...
def load_theme(ini_path: str = '', def_theme: GuiTheme = ...) -> GuiTheme: ...
def load_settings(
  ini_path: str = '', def_theme: GuiTheme = ...
) -> GuiTheme: ...
def save_settings(theme: GuiTheme, ini_path: str = '') -> None: ...
def reset_config(ini_path: str = '') -> None: ...
def label_selection_gui(vis_options: MjvOption) -> bool: ...
def frame_selection_gui(vis_options: MjvOption) -> bool: ...
def camera_selection_gui(
  model: MjModel, data: MjData, camera: MjvCamera, ux_state: UxState
) -> bool: ...
def set_camera(model: MjModel, camera: MjvCamera, request_idx: int) -> int: ...
def set_speed_index(
  step_control: StepControl, speed_index: int, request_idx: int
) -> int: ...
def physics_gui(
  model: MjModel, spec: MjSpec | None = None, min_width: float = 150.0
) -> None: ...
def rendering_gui(
  model: MjModel, vis_options: MjvOption, render_flags: RenderFlags
) -> None: ...
def groups_gui(
  model: MjModel, vis_options: MjvOption, min_width: float = 150.0
) -> None: ...
def visualization_gui(
  model: MjModel,
  vis_options: MjvOption,
  camera: MjvCamera,
  min_width: float = 150.0,
) -> None: ...
def controls_gui(
  model: MjModel, data: MjData, vis_options: MjvOption
) -> None: ...
def joints_gui(
  model: MjModel, data: MjData, vis_options: MjvOption
) -> None: ...
def sensor_gui(model: MjModel, data: MjData) -> None: ...
def state_gui(
  model: MjModel, data: MjData, ux_state: UxState, min_width: float = 150.0
) -> None: ...
def watch_gui(model: MjModel, data: MjData, ux_state: UxState) -> None: ...
def noise_gui(step_control: StepControl) -> None: ...
def convergence_gui(model: MjModel, data: MjData) -> None: ...
def counts_gui(model: MjModel, data: MjData) -> None: ...
def info_gui(
  model: MjModel, data: MjData, paused: bool, fps: float
) -> None: ...

FREE_CAMERA_IDX: int
TUMBLE_CAMERA_IDX: int
TRACKING_CAMERA_IDX: int

class CameraMotion:
  __members__: ClassVar[dict[str, CameraMotion]]
  ZOOM: ClassVar[CameraMotion]
  ORBIT: ClassVar[CameraMotion]
  TRUCK_PEDESTAL: ClassVar[CameraMotion]
  TRUCK_DOLLY: ClassVar[CameraMotion]
  PAN_TILT: ClassVar[CameraMotion]
  PLANAR_MOVE_H: ClassVar[CameraMotion]
  PLANAR_MOVE_V: ClassVar[CameraMotion]
  def __init__(self, value: int) -> None: ...
  def __int__(self) -> int: ...
  def __index__(self) -> int: ...
  @property
  def name(self) -> str: ...
  @property
  def value(self) -> int: ...

ZOOM: CameraMotion
ORBIT: CameraMotion
TRUCK_PEDESTAL: CameraMotion
TRUCK_DOLLY: CameraMotion
PAN_TILT: CameraMotion
PLANAR_MOVE_H: CameraMotion
PLANAR_MOVE_V: CameraMotion

def MoveCamera(
  model: MjModel,
  data: MjData,
  cam: MjvCamera,
  motion: CameraMotion,
  dx: float,
  dy: float,
) -> None: ...
def InitPerturb(
  model: MjModel, data: MjData, cam: MjvCamera, pert: MjvPerturb, active: int
) -> None: ...
def MovePerturb(
  model: MjModel,
  data: MjData,
  cam: MjvCamera,
  pert: MjvPerturb,
  action: int,
  reldx: float,
  reldy: float,
) -> None: ...

class PickResult:
  """A pick hit; point is an exact three-coordinate world-space tuple."""

  dist: float
  body: int
  geom: int
  flex: int
  skin: int
  @property
  def point(self) -> tuple[float, float, float]: ...
  @point.setter
  def point(self, value: tuple[float, float, float]) -> None: ...

def Pick(
  model: MjModel,
  data: MjData,
  cam: MjvCamera,
  x: float,
  y: float,
  aspect_ratio: float,
  opt: MjvOption,
) -> PickResult: ...
def camera_to_string(data: MjData, camera: MjvCamera) -> str: ...

class PipSource:
  def __init__(
    self, name: str = '', texture: int = ..., aspect_ratio: float = 1.0
  ) -> None: ...
  name: str
  texture: int
  aspect_ratio: float

class PipState:
  def __init__(self) -> None: ...
  camera: int
  texture: int

def pip_gui(
  sources: Sequence[PipSource], pips: Sequence[PipState]
) -> list[PipState]: ...
