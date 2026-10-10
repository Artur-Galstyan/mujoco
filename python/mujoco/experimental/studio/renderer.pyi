# Copyright 2026 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS.

from mujoco import MjData, MjModel, MjvCamera, MjvOption, MjvPerturb

class Renderer:
  """Standalone Filament renderer; render produces packed RGB bytes."""

  def __init__(self, graphics_mode_str: str, /) -> None: ...
  def Init(self, model: MjModel, /) -> None: ...
  def Render(
    self,
    model: MjModel,
    data: MjData,
    perturb: MjvPerturb | None,
    camera: MjvCamera | None,
    vis_option: MjvOption | None,
    width: int,
    height: int,
    /,
  ) -> bytes:
    """Render packed RGB bytes; allocate width * height * 3 bytes."""
  def get_render_flags(self) -> memoryview:
    """Get a writable, one-dimensional view of the renderer's flag bytes."""
