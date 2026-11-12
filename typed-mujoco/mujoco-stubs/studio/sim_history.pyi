# Copyright 2026 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS.
"""Simulation history recording and scrubbing plugin."""

from dataclasses import dataclass

from mujoco.studio import messages, plugin, sim, ux

@dataclass(frozen=True)
class SimHistorySnapshot(messages.Snapshot):
  """Bidirectional snapshot carrying history-buffer state."""

  size: int = 0
  head_time: float = 0.0
  tail_time: float = 0.0
  index: int = 0
  @classmethod
  def from_ux_state(cls, ux_state: ux.UxState) -> SimHistorySnapshot | None:
    """Pack the current scrub index of ``ux_state`` into a snapshot."""

class SimHistory(plugin.SimPlugin):
  """Sim-side plugin for simulation history recording and scrubbing."""

  sim_history: sim.SimHistory
  def __init__(self, *, min_seconds: float = 0.0) -> None: ...
  @property
  def frame_interval(self) -> float:
    """Simulated seconds between recorded frames for the current model."""
