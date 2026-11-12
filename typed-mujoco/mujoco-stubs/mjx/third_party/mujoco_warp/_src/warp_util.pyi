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

"""Warp utilities: event tracing and kernel caching."""

from collections.abc import Callable
from types import TracebackType
from typing import ParamSpec, Self, TypeAlias, TypeVar

_P = ParamSpec('_P')
_R = TypeVar('_R')

_EventTrace: TypeAlias = dict[str, tuple[tuple[float, ...], _EventTrace]]

def is_array_spec(typ: object) -> bool: ...
def mark_batched(obj: object) -> None: ...

class EventTracer:
  """Calculates elapsed times of functions annotated with `event_scope`."""

  def __init__(self, enabled: bool = ...) -> None: ...
  def __enter__(self) -> Self: ...
  def trace(self) -> _EventTrace: ...
  def __exit__(
    self,
    type: type[BaseException] | None,
    value: BaseException | None,
    traceback: TracebackType | None,
  ) -> None: ...

def event_scope(fn: Callable[_P, _R], name: str = ...) -> Callable[_P, _R]: ...
def cache_kernel(func: Callable[_P, _R]) -> Callable[_P, _R]: ...
def check_toolkit_driver() -> None: ...
