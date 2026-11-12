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
"""Forward step functions."""

from collections.abc import Callable
from typing import ParamSpec, TypeVar

from mujoco.mjx._src.types import Data, Model

_P = ParamSpec('_P')
_R = TypeVar('_R')

def named_scope(fn: Callable[_P, _R], name: str = '') -> Callable[_P, _R]: ...
@named_scope
def fwd_position(m: Model, d: Data) -> Data: ...
@named_scope
def fwd_velocity(m: Model, d: Data) -> Data: ...
@named_scope
def fwd_actuation(m: Model, d: Data) -> Data: ...
@named_scope
def fwd_acceleration(m: Model, d: Data) -> Data: ...
@named_scope
def euler(m: Model, d: Data) -> Data: ...
@named_scope
def rungekutta4(m: Model, d: Data) -> Data: ...
@named_scope
def implicit(m: Model, d: Data) -> Data: ...
@named_scope
def forward(m: Model, d: Data) -> Data: ...
@named_scope
def step(m: Model, d: Data) -> Data: ...
