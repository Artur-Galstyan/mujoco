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
"""Inverse dynamics."""

import warp as wp

from mujoco.mjx.third_party.mujoco_warp._src.types import (
  Data,
  Model,
)

def discrete_acc(m: Model, d: Data, qacc: wp.array[float]) -> None:
  """Convert discrete-time qacc to continuous-time qacc."""

def inv_constraint(m: Model, d: Data) -> None:
  """Inverse constraint solver."""

def inverse(m: Model, d: Data) -> None:
  """Inverse dynamics."""
