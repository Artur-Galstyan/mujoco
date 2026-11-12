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
"""Base classes for Studio plugins."""

import os
from collections.abc import Iterable
from typing import TypeAlias

from mujoco.studio import viewer_handle, viewer_protocol

SimInitEvent = viewer_handle.SimInitEvent
ViewerInitEvent = viewer_protocol.ViewerInitEvent
ThemeEvent = viewer_protocol.ThemeEvent

PLUGIN_PACKAGE: str

class SimPlugin:
  """Base class for simulation-side plugins."""

  _handle: viewer_handle.ViewerHandle | None
  @property
  def handle(self) -> viewer_handle.ViewerHandle: ...
  @handle.setter
  def handle(self, value: viewer_handle.ViewerHandle | None) -> None: ...

class ViewerPlugin:
  """Base class for viewer-side plugins."""

  _viewer: viewer_protocol.Viewer | None
  @property
  def viewer(self) -> viewer_protocol.Viewer: ...
  @viewer.setter
  def viewer(self, value: viewer_protocol.Viewer | None) -> None: ...

Plugin: TypeAlias = SimPlugin | ViewerPlugin

def split_plugins(
  plugins: Iterable[Plugin] | None = None,
) -> tuple[list[ViewerPlugin], list[SimPlugin]]:
  """Split plugins into a tuple of viewer plugins and sim plugins."""

def load_plugins(
  paths: Iterable[str | os.PathLike[str]] | None = None,
) -> list[Plugin]:
  """Import `.py` files and instantiate every plugin class in them."""
