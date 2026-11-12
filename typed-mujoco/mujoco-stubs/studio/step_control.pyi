# Copyright 2026 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS.
"""Default sim-side stepping plugin."""

from mujoco.studio import plugin, sim

class StepControl(plugin.SimPlugin):
  """Default sim plugin that steps CPU physics with real-time pacing."""

  step_control: sim.StepControl
  def __init__(self) -> None: ...
