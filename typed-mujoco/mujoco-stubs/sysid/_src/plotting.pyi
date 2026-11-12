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

"""Plotting utilities."""

from collections.abc import Sequence

import numpy as np
from numpy import typing as npt

import mujoco

def render_rollout(
  model: mujoco.MjModel | Sequence[mujoco.MjModel],
  data: mujoco.MjData,
  state: npt.NDArray[np.float64],
  framerate: int,
  camera: str | int = -1,
  width: int = 640,
  height: int = 480,
  light_pos: Sequence[float] | None = None,
) -> list[npt.NDArray[np.uint8]]:
  """Renders a rollout or batch of rollouts."""
