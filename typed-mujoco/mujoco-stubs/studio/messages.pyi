# Copyright 2026 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS.
"""Messages and channels for Studio."""

import enum
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (
  Any,
  Protocol,
  TypeAlias,
  TypeVar,
  overload,
  runtime_checkable,
)

import numpy as np
from numpy import typing as npt

import mujoco
from mujoco.studio import sim

_F = TypeVar('_F', bound=Callable[..., Any])

KEYFRAME_STATE_SIG: int

class Message:
  """Base class for data sent between the sim and the viewer."""

class Snapshot(Message):
  """Base class for latest-wins, droppable messages."""

class Event(Message):
  """Base class for reliable, ordered, never dropped messages."""

@runtime_checkable
class SnapshotChannel(Protocol):
  """Channel to transport Snapshot messages with latest-wins semantics."""

  def put(self, value: Snapshot) -> None:
    """Overwrite the latest snapshot of the same type."""
  def get(self) -> list[Snapshot]:
    """Return the latest pending snapshots of each type."""
  def reclaim_unconsumed(
    self, snapshot_type: type[Snapshot]
  ) -> Snapshot | None:
    """Pop and return an unconsumed pending snapshot of the given type."""
  def close(self) -> None:
    """Release resources held by the channel."""

@runtime_checkable
class EventChannel(Protocol):
  """Channel to transport Event messages in an ordered stream."""

  def put(self, value: Event) -> None:
    """Append an event to the stream."""
  def get(self) -> list[Event]:
    """Return all pending events, in order."""
  def close(self) -> None:
    """Release resources held by the channel."""

MakeMessage: TypeAlias = Callable[[], Message | None]

@contextmanager
def send_if_changes(
  send_fn: Callable[[Message], None],
  *make_messages: MakeMessage,
  should_send: Callable[[], bool] | None = None,
) -> Generator[None]:
  """Send the messages that change during the ``with`` block."""

@dataclass(frozen=True, eq=False)
class StateSnapshot(Snapshot):
  """A snapshot message that transports a MuJoCo state."""

  state: npt.NDArray[np.float64]
  state_sig: int
  def __eq__(self, other: object) -> bool: ...

@dataclass(frozen=True, eq=False)
class PerturbSnapshot(Snapshot):
  """Carries perturbation forces from the viewer to the simulation."""

  state: npt.NDArray[np.float64]
  state_sig: int
  def __eq__(self, other: object) -> bool: ...

@dataclass(frozen=True, eq=False)
class StateEvent(Event):
  """A reliable event that transports a MuJoCo state."""

  state: npt.NDArray[np.float64]
  state_sig: int
  @classmethod
  def from_data(
    cls, model: mujoco.MjModel, data: mujoco.MjData, state_sig: int
  ) -> StateEvent | None:
    """Pack the selected components of ``data`` into a ``StateEvent``."""
  def __eq__(self, other: object) -> bool: ...

@dataclass(frozen=True)
class ModelEvent(Event):
  """An event that transports a MuJoCo model."""

  model: mujoco.MjModel
  path: str = ''

@dataclass(frozen=True)
class LoadModelEvent(Event):
  """Ask the sim to load a model file, or to reload the current model."""

  path: str = ''

@dataclass(frozen=True)
class PreloadModelEvent(Event):
  """A hint that ``model`` may become the current model soon."""

  model: mujoco.MjModel

@dataclass(frozen=True)
class PostModelEvent(Event):
  """Lifecycle event dispatched locally after a new model and data load."""

  model: mujoco.MjModel
  data: mujoco.MjData
  path: str = ''

@dataclass(frozen=True)
class SingleStepEvent(Event):
  """Sent from viewer to sim when the user requests a single step."""

@dataclass(frozen=True)
class RequestPauseEvent(Event):
  """Request a pause state change on the sim or the viewer."""

  pause_state: sim.PauseState

@dataclass(frozen=True)
class ResetEvent(Event):
  """An event requesting to reset the simulation."""

  key: int | None = None

@dataclass(frozen=True, eq=False)
class KeyframeEvent(Event):
  """Carries a keyframe saved in the viewer to the simulation model."""

  key: int
  state: npt.NDArray[np.float64]
  @classmethod
  def from_model(cls, model: mujoco.MjModel, key: int) -> KeyframeEvent | None:
    """Pack keyframe ``key`` of ``model`` into a ``KeyframeEvent``."""
  def __eq__(self, other: object) -> bool: ...

@dataclass(frozen=True)
class ThreadPoolEvent(Event):
  """An event requesting to resize the simulation's thread pool."""

  nthread: int

@dataclass(frozen=True)
class PreStepEvent(Event):
  """Lifecycle event dispatched by ViewerHandle.sync before StepEvent."""

  model: mujoco.MjModel
  data: mujoco.MjData

@dataclass(frozen=True)
class StepEvent(Event):
  """Lifecycle event dispatched by ViewerHandle.sync after PreStepEvent."""

  model: mujoco.MjModel
  data: mujoco.MjData

@dataclass(frozen=True)
class PostStepEvent(Event):
  """Lifecycle event dispatched by ViewerHandle.sync after StepEvent."""

  model: mujoco.MjModel
  data: mujoco.MjData

@dataclass(frozen=True)
class UpdateEvent(Event):
  """Lifecycle event dispatched on every viewer frame before building GUI."""

@dataclass(frozen=True)
class BuildGuiEvent(Event):
  """Lifecycle event dispatched on every viewer frame to build ImGui."""

@dataclass(frozen=True)
class MjOptionSnapshot(Snapshot):
  """Bidirectional snapshot carrying mjOption state."""

  opt: mujoco.MjOption

@dataclass(frozen=True)
class StepControlSnapshot(Snapshot):
  """Bidirectional snapshot carrying step control state."""

  pause_state: sim.PauseState = ...
  speed: float = 100.0
  noise_scale: float = 0.0
  noise_rate: float = 0.0
  speed_measured: float = -1.0
  @classmethod
  def from_step_control(
    cls, step_control: sim.StepControl, speed_measured: float = -1.0
  ) -> StepControlSnapshot:
    """Pack the settings of ``step_control`` into a snapshot."""

@dataclass(frozen=True)
class ExitEvent(Event):
  """An event requesting to exit."""

  reason: str = ''

def describe_exception(ex: BaseException | None) -> str:
  """Format an error exception as a single line."""

@dataclass(frozen=True)
class HotReloadEvent(Event):
  """Dispatched after plugins are hot-reloaded."""

  plugins: tuple[Any, ...]

class Priority(enum.IntEnum):
  """Execution priority levels for message handlers."""

  INTERNAL = 1
  LIBRARY = 10
  USER = 100
  CRITICAL = 1000

@overload
def handler(fn: _F, *, priority: int = ...) -> _F: ...
@overload
def handler(fn: None = None, *, priority: int = ...) -> Callable[[_F], _F]: ...
