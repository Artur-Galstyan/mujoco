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
"""Opt-in hot reloading of plugin modules."""

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from mujoco.studio import hot_reload_utils as utils
from mujoco.studio import plugin_registry, viewer_handle, viewer_protocol

FileWatcher = utils.FileWatcher

logger: logging.Logger

@dataclass(frozen=True)
class Options:
  """What hot reloading covers."""

  model: bool = True
  plugins: bool | frozenset[str] = True
  @classmethod
  def parse(cls, text: str) -> Options:
    """Parse a `--hot_reload` value."""
  def reloads(self, plugin: Any) -> bool:
    """Return whether a plugin is reloaded when its module is saved."""

class Reloader:
  """Watch plugin modules and hot-reload them into plugin registries."""

  generation: int
  def __init__(
    self, options: Options = ..., poll_interval: float = ...
  ) -> None: ...
  @property
  def options(self) -> Options: ...
  @property
  def watched_modules(self) -> set[str]: ...
  @property
  def poll_interval(self) -> float: ...
  def tick(
    self,
    registry: plugin_registry.PluginRegistry,
    host: viewer_protocol.Viewer | viewer_handle.ViewerHandle,
  ) -> None:
    """Reload changed modules and update the plugins of a registry."""

def enable(what: str = 'all') -> None:
  """Configure hot reloading for this process; an empty string disables it."""

def watches_model() -> bool:
  """Return whether the shown model file is reloaded when saved."""

def watch_files(paths: Iterable[str]) -> FileWatcher | None:
  """Return a watcher for the files a model is built from, if enabled."""

def tick(
  registry: plugin_registry.PluginRegistry,
  host: viewer_protocol.Viewer | viewer_handle.ViewerHandle,
) -> None:
  """Hot-reload changed plugin modules into a registry, if enabled."""
