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
"""In-process threaded viewer launcher functionality and channels."""

from collections.abc import Callable
from typing import Any

from mujoco.studio import endpoints, messages, viewer_handle

class ThreadSnapshotChannel(messages.SnapshotChannel):
  """SnapshotChannel for in-process threaded mode."""

  def __init__(self) -> None: ...
  def put(  # ty: ignore[invalid-method-override]
    self, snapshot: messages.Snapshot
  ) -> None: ...
  def reclaim_unconsumed(
    self, snapshot_type: type[messages.Snapshot]
  ) -> messages.Snapshot | None:
    """Pop an unconsumed pending snapshot of the given type."""
  def get(self) -> list[messages.Snapshot]: ...
  def close(self) -> None: ...

class ThreadEventChannel(messages.EventChannel):
  """EventChannel for in-process threaded mode using a thread-safe queue."""

  def __init__(self) -> None: ...
  def put(  # ty: ignore[invalid-method-override]
    self, event: messages.Event
  ) -> None: ...
  def get(self) -> list[messages.Event]: ...
  def close(self) -> None: ...

def make_thread_endpoints() -> tuple[
  endpoints.ViewerEndpoint, endpoints.SimEndpoint
]:
  """Create connected in-process thread-safe endpoints."""

def launch_thread(
  target_fn: Callable[[endpoints.ViewerEndpoint], None],
  *,
  sim_plugins: list[Any] | None = None,
) -> viewer_handle.ViewerHandle:
  """Launch a viewer target function in a daemon thread."""
