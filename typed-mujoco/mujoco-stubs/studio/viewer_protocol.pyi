# Copyright 2026 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS.
"""Base class and configuration for any viewer."""

import abc
import logging
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Literal

import mujoco
from mujoco.studio import endpoints, messages, plugin_registry, ux
from mujoco.studio.web.client_camera import ClientCameraSync

logger: logging.Logger

GFX_MODES: tuple[
  Literal['opengl'],
  Literal['opengl_headless'],
  Literal['opengl_software'],
  Literal['vulkan'],
  Literal['vulkan_software'],
  Literal['web'],
  Literal['webgl'],
]

@dataclass
class ViewerConfig:
  """Common configuration for creating a viewer window."""

  title: str = ''
  width: int = 1200
  height: int = 800
  gfx: str = ''
  http_port: int = 0
  open_browser: bool = False
  hot_reload: str = ''

@dataclass(frozen=True)
class ViewerInitEvent(messages.Event):
  """Lifecycle event dispatched locally once when a Viewer initializes."""

  viewer: Viewer

@dataclass(frozen=True)
class ThemeEvent(messages.Event):
  """Dispatched on the viewer when the host changes the UI theme."""

  theme: ux.GuiTheme

class Viewer(abc.ABC):
  """Base class for any viewer."""

  config: ViewerConfig
  exit_reason: str
  model: mujoco.MjModel
  data: mujoco.MjData
  model_path: str
  camera: mujoco.MjvCamera
  cam_speed: float
  client_camera: ClientCameraSync | None
  camera_index: int
  perturb: mujoco.MjvPerturb
  vis_options: mujoco.MjvOption
  extra_geoms: list[mujoco.MjvGeom]
  render_flags: ux.RenderFlags
  plugins: plugin_registry.PluginRegistry
  def __init__(
    self,
    config: ViewerConfig,
    endpoint: endpoints.ViewerEndpoint,
    *,
    model: mujoco.MjModel | None = None,
    plugins: list[Any] | None = None,
    camera: mujoco.MjvCamera | None = None,
    vis_options: mujoco.MjvOption | None = None,
    perturb: mujoco.MjvPerturb | None = None,
    render_flags: ux.RenderFlags | None = None,
    extra_geoms: list[mujoco.MjvGeom] | None = None,
  ) -> None:
    """Initialize the Viewer."""
  def close(self) -> None:
    """Close the viewer, send an exit event and shut down the endpoint."""
  def is_running(self) -> bool:
    """Return True while the viewer has not been closed."""
  def send_to_sim(self, message: messages.Message) -> None:
    """Send a message to the simulation process."""
  def send_if_changes(
    self, *make_messages: messages.MakeMessage
  ) -> AbstractContextManager[None]:
    """Send the messages that change during the ``with`` block to the sim."""
  def get_sim_events(self) -> list[messages.Event]:
    """Return all pending events from the simulation."""
  def get_sim_snapshots(self) -> list[messages.Snapshot]:
    """Return all pending latest snapshots from the simulation."""
  def dispatch(self, message: messages.Message) -> None:
    """Dispatch a message to registered handlers in priority order."""
  def load_model(self, model: mujoco.MjModel, model_path: str = '') -> None:
    """Deep-copy a model and create fresh data for the viewer."""
  @abc.abstractmethod
  def prepare_next_frame(self) -> bool:
    """Advance to the next frame; return whether one is ready to render."""
  @abc.abstractmethod
  def sync(self) -> None:
    """Render the scene using the viewer's current model and data."""
  def sync_idle(self) -> None:
    """Update viewer state when prepare_next_frame returned False."""
  @abc.abstractmethod
  def get_drop_file(self) -> str: ...
  @abc.abstractmethod
  def upload_image(
    self, tex_id: int, img: str | bytes, width: int, height: int, bpp: int
  ) -> int: ...

def run_viewer_loop(viewer: Viewer) -> None:
  """Process sim messages, dispatch lifecycle events and sync until exit."""
