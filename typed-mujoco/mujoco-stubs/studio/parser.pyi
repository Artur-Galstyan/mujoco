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
"""Load and compile MuJoCo models for the studio viewer."""

import mujoco

def load(filepath: str) -> mujoco.MjModel:
  """Load and compile a model file: XML, URDF, MJB, or MJZ/ZIP."""

def parse(filepath: str) -> mujoco.MjData:
  """Load a model file and return a fresh `MjData` for it."""

def dependencies(filepath: str) -> list[str]:
  """Return the files a model file is built from, including the file."""
