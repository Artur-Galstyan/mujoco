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
"""MJX Warp integration: imports of warp and mujoco_warp."""

import warp as warp

from mujoco.mjx.third_party import mujoco_warp as mujoco_warp
from mujoco.mjx.third_party.mujoco_warp._src import io as _mjwp_io
from mujoco.mjx.third_party.mujoco_warp._src import types as _mjwp_types
from mujoco.mjx.warp import render_context as render_context
from mujoco.mjx.warp import types as types

WARP_INSTALLED: bool
mjwp_io = _mjwp_io
mjwp_types = _mjwp_types
