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
"""Endpoints route messages to the right channel."""

from mujoco.studio import messages as vp

class ViewerEndpoint:
  """Viewer-side endpoint that routes Messages to and from the Sim."""

  def __init__(
    self,
    *,
    s2v_snapshot: vp.SnapshotChannel,
    s2v_events: vp.EventChannel,
    v2s_events: vp.EventChannel,
    v2s_snapshot: vp.SnapshotChannel,
  ) -> None: ...
  def send_to_sim(self, message: vp.Message) -> None: ...
  def get_sim_events(self) -> list[vp.Event]:
    """Return all pending events from the simulation."""
  def get_sim_snapshots(self) -> list[vp.Snapshot]:
    """Return all pending latest snapshots from the simulation."""
  def close(self) -> None:
    """Close all channels owned by this endpoint."""

class SimEndpoint:
  """Sim-side endpoint that routes Messages to and from the Viewer."""

  def __init__(
    self,
    *,
    s2v_snapshot: vp.SnapshotChannel,
    s2v_events: vp.EventChannel,
    v2s_events: vp.EventChannel,
    v2s_snapshot: vp.SnapshotChannel,
  ) -> None: ...
  def send_to_viewer(self, message: vp.Message) -> None: ...
  def get_viewer_events(self) -> list[vp.Event]:
    """Return all pending events from the viewer."""
  def get_viewer_snapshots(self) -> list[vp.Snapshot]:
    """Return all pending latest snapshots from the viewer."""
  def reclaim_to_viewer(
    self, snapshot_type: type[vp.Snapshot]
  ) -> vp.Snapshot | None:
    """Reclaim an unconsumed outgoing snapshot of the given type."""
  def close(self) -> None:
    """Close all channels owned by this endpoint."""

def make_endpoints(
  *,
  s2v_snapshot: vp.SnapshotChannel,
  s2v_events: vp.EventChannel,
  v2s_events: vp.EventChannel,
  v2s_snapshot: vp.SnapshotChannel,
) -> tuple[ViewerEndpoint, SimEndpoint]:
  """Return viewer and simulation endpoints."""
