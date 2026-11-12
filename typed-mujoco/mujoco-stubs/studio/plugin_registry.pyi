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
"""Discover plugin `@handler` methods and dispatch messages to them."""

from collections.abc import Callable
from typing import Any, NamedTuple, TypeAlias

from mujoco.studio import messages

Handler: TypeAlias = Callable[..., Any]
HandlerWrapper: TypeAlias = Callable[[Any, Handler], Handler]

class PrioritizedHandler(NamedTuple):
  """A registered handler and its priority; higher priorities run first."""

  priority: int
  handler: Handler

class PluginRegistry:
  """Runtime registry of discovered handlers from plugin instances."""

  plugins: list[Any]
  handlers: dict[
    type[messages.Message] | tuple[str, str], list[PrioritizedHandler]
  ]
  def __init__(self, plugins: list[Any] | None = None) -> None: ...
  def rebuild_handlers(self, wrap: HandlerWrapper | None = None) -> None:
    """Discover and register handlers for all plugins."""
  def dispatch(self, message: messages.Message) -> None:
    """Dispatch a message to registered handlers in priority order."""
