# Copyright 2026 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS.
"""Hands the controller's camera between the browser and the web viewer."""

import mujoco

class ClientCameraSync:
  """Tracks the viewer's camera generation."""

  generation: int
  def __init__(self) -> None: ...
  def apply(self, camera: mujoco.MjvCamera, message: bytes) -> bool:
    """Adopt a camera the browser sent, unless it predates ``generation``."""
  def update_generation(self, camera: mujoco.MjvCamera) -> int:
    """Return the generation to stream with ``camera``."""
