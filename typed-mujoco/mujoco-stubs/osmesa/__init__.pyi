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

"""An OSMesa context for software-based OpenGL rendering."""

import contextlib
from collections.abc import Generator

PYOPENGL_PLATFORM: str | None

class GLContext:
  """An OSMesa context for software-based OpenGL rendering."""

  def __init__(self, max_width: int, max_height: int) -> None: ...
  def make_current(self) -> None: ...
  @contextlib.contextmanager
  def as_current(self) -> Generator[None]:
    """Temporarily makes this context current, restoring any prior context."""
  def free(self) -> None:
    """Frees resources associated with this context."""
  def __del__(self) -> None: ...
