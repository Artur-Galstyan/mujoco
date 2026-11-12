# Copyright 2026 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS.
"""Viewer handle and event handlers for the simulation side."""

import logging
import os
from collections.abc import Callable, Iterable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from types import TracebackType
from typing import Any, TypeAlias

from typing_extensions import Self

import mujoco
from mujoco.studio import endpoints, messages

logger: logging.Logger

IsAliveFn: TypeAlias = Callable[[], bool]
ShutdownFn: TypeAlias = Callable[[float], None]
ForwardErrorFn: TypeAlias = Callable[[], None]

@dataclass(frozen=True)
class SimInitEvent(messages.Event):
  """Lifecycle event dispatched locally once when ViewerHandle initializes."""

  handle: ViewerHandle

class ViewerHandle:
  """A handle for interacting with a running viewer from the sim."""

  model: mujoco.MjModel | None
  data: mujoco.MjData | None
  model_path: str
  def __init__(
    self,
    sim_endpoint: endpoints.SimEndpoint,
    *,
    sim_plugins: list[Any] | None = None,
    is_alive_fn: IsAliveFn | None = None,
    shutdown_fn: ShutdownFn | None = None,
    forward_error_fn: ForwardErrorFn | None = None,
  ) -> None: ...
  def close(self, reason: str = '') -> None:
    """Signal the viewer to exit and wait for it to shut down."""
  def __enter__(self) -> Self: ...
  def __exit__(
    self,
    exc_type: type[BaseException] | None,
    exc_val: BaseException | None,
    exc_tb: TracebackType | None,
  ) -> None: ...
  def is_running(self) -> bool:
    """Return True while the viewer is open."""
  def send_to_viewer(self, message: messages.Message) -> None:
    """Send an event or snapshot message to the viewer process."""
  def send_if_changes(
    self, *make_messages: messages.MakeMessage
  ) -> AbstractContextManager[None]:
    """Send the messages that change during the ``with`` block."""
  def dispatch(self, message: messages.Message) -> None:
    """Dispatch a message to sim-side handlers in priority order."""
  def set_model(
    self,
    model: mujoco.MjSpec | mujoco.MjModel,
    path: str = '',
    data: mujoco.MjData | None = None,
    *,
    watch: Iterable[str | os.PathLike[str]] = (),
  ) -> None:
    """Publish ``model`` to sim plugins and the viewer."""
  def reload(self) -> None:
    """Reload the current model, as the Reload button does."""
  def sync(
    self,
    model: mujoco.MjModel | None = None,
    data: mujoco.MjData | None = None,
  ) -> tuple[mujoco.MjModel | None, mujoco.MjData | None]:
    """Sync the simulation with the viewer and return the sim state."""

def run_sim_loop(handle: ViewerHandle) -> None:
  """Run the standard simulation sync loop until the viewer closes."""
