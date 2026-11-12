# Copyright 2026 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS.
"""Utility functions for Studio viewer implementations."""

import mujoco
from mujoco.studio import viewer_protocol

def send_state(
  viewer: viewer_protocol.Viewer,
  model: mujoco.MjModel,
  data: mujoco.MjData,
  state_sig: int,
) -> None:
  """Send the selected components of the viewer's state to the sim thread."""

def apply_perturb(
  viewer: viewer_protocol.Viewer,
  model: mujoco.MjModel | None,
  data: mujoco.MjData | None,
  is_paused: bool = False,
) -> None:
  """Apply perturbation to the model and send the forces to the sim thread."""
