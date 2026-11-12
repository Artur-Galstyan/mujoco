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
"""File-watching, module-graph, and class helpers for hot_reload."""

import types
from collections.abc import Iterable, Mapping
from typing import Any

class FileWatcher:
  """Detect file changes by polling `os.stat`."""

  def __init__(
    self, paths: Iterable[str] = (), poll_interval: float = 0.0
  ) -> None: ...
  def watch(self, path: str) -> str:
    """Start watching a path and return the resolved path."""
  def poll(self) -> set[str]:
    """Return the resolved paths that changed since the previous poll."""

def source_path(module: types.ModuleType) -> str | None:
  """Return the resolved `.py` source of a module, or None."""

def module_dir(name: str) -> str | None:
  """Return the directory of a loaded module's source, if it has one."""

def local_imports(module: types.ModuleType) -> list[types.ModuleType]:
  """Return the `.py` modules in the same directory that a module uses."""

def reload_module(module: types.ModuleType) -> None:
  """Re-execute the current source of a module in place."""

def reload_order(
  deps: Mapping[str, set[str]], roots: Iterable[str]
) -> list[str]:
  """Return the roots and their importers, dependencies first."""

def current_class(cls: type[Any]) -> type[Any] | None:
  """Return the class now bound to the module and name of a class."""

def class_name(cls: type[Any]) -> str:
  """Return the fully qualified name of a class."""

def same_default(a: Any, b: Any) -> bool:
  """Return whether two class-attribute default values are equal."""
