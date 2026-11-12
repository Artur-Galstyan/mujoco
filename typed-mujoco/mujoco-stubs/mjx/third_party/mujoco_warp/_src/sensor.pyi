# Copyright 2025 The Newton Developers
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
"""Sensor computation and energy."""

from mujoco.mjx.third_party.mujoco_warp._src.types import (
  Data,
  Model,
)

def sensor_pos(m: Model, d: Data) -> None:
  """Compute position-dependent sensor values."""

def sensor_vel(m: Model, d: Data) -> None:
  """Compute velocity-dependent sensor values."""

def sensor_acc(m: Model, d: Data, skip_rne_postconstraint: bool = ...) -> None:
  """Compute acceleration-dependent sensor values."""

def energy_pos(m: Model, d: Data) -> None:
  """Position-dependent energy (potential)."""

def energy_vel(m: Model, d: Data) -> None:
  """Velocity-dependent energy (kinetic)."""
