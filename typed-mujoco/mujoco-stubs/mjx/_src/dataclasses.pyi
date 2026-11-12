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
"""Wrapper that automatically registers dataclass as a Jax PyTree."""

import dataclasses
from typing import Any, TypeVar

import jax
from typing_extensions import Self, dataclass_transform

_T = TypeVar('_T')
TNode = TypeVar('TNode', bound=PyTreeNode)  # noqa: PYI001

def dataclass(clz: _T, register_as_pytree: bool) -> _T: ...

@dataclass_transform(frozen_default=True)
class PyTreeNode:
  """Base class for dataclasses that should act like a JAX pytree node."""

  def __init_subclass__(
    cls, register_as_pytree: bool = ..., **kwargs: Any
  ) -> None: ...
  def __init__(self, *args: Any, **kwargs: Any) -> None: ...
  def replace(self, **overrides: Any) -> Self: ...
  @classmethod
  def fields(cls) -> tuple[dataclasses.Field[Any], ...]: ...
  def tree_replace(
    self, params: dict[str, jax.typing.ArrayLike | None]
  ) -> Self: ...
