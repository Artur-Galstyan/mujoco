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
"""Scan across data ordered by body joint types and kinematic tree order."""

from collections.abc import Callable
from typing import TypeVar

from mujoco.mjx._src.types import Model

Y = TypeVar('Y')  # noqa: PYI001

def flat(
  m: Model,
  f: Callable[..., Y],
  in_types: str,
  out_types: str,
  *args: object,
  group_by: str = 'j',
) -> Y: ...
def body_tree(
  m: Model,
  f: Callable[..., Y],
  in_types: str,
  out_types: str,
  *args: object,
  reverse: bool = False,
) -> Y: ...
