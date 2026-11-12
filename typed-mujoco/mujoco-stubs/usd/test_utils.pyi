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
"""Test scenes for the USD tests."""

import dataclasses
from typing import Any, TypeAlias

import mujoco

_Stage: TypeAlias = Any

@dataclasses.dataclass
class TestData:
  stage: _Stage
  spec: mujoco.MjSpec
  model: mujoco.MjModel
  data: mujoco.MjData

def simple_falling_cube() -> TestData: ...
def simple_nested_hierarchy() -> TestData: ...
