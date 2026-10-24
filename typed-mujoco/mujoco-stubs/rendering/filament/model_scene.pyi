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
"""Model-backed lights and renderables for a Filament scene."""

from mujoco import MjData, MjModel, mjrf, mjtObj

class ModelScene:
  def __init__(
    self, ctx: mjrf.Context, model: MjModel, scene: mjrf.Scene | None = None
  ) -> None: ...
  def update(self, data: MjData) -> None: ...
  @property
  def scene(self) -> mjrf.Scene: ...
  def get_light(self, index: int) -> mjrf.Light | None:
    """Return the model light, or None when the index has no light."""
  def get_renderable(
    self, obj_type: mjtObj, obj_index: int, *, sub_index: int = -1
  ) -> mjrf.Renderable | None:
    """Return the model renderable, or None when there is no match."""
