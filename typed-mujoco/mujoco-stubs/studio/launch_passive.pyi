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
"""Unified non-blocking viewer launcher."""

from collections.abc import Iterable

import mujoco
from mujoco.studio import plugin, viewer_handle, viewer_protocol

def launch(
  config: viewer_protocol.ViewerConfig,
  *,
  plugins: Iterable[plugin.Plugin] | None = None,
) -> viewer_handle.ViewerHandle:
  """Launch the viewer in a daemon thread without blocking."""

def run(
  config: viewer_protocol.ViewerConfig,
  *,
  model: mujoco.MjModel | None = None,
  data: mujoco.MjData | None = None,
  model_path: str | None = None,
  plugins: Iterable[plugin.Plugin] | None = None,
) -> None:
  """Launch the viewer and run the blocking simulation sync loop."""
