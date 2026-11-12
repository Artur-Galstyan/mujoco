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
"""CLI for converting legacy MSH files to Wavefront OBJ files."""

import dataclasses
import pathlib

import numpy as np
from numpy import typing as npt

@dataclasses.dataclass(frozen=True)
class Msh:
  """MuJoCo legacy binary msh file."""

  vertex_positions: npt.NDArray[np.float32]
  vertex_normals: npt.NDArray[np.float32]
  vertex_texcoords: npt.NDArray[np.float32]
  face_vertex_indices: npt.NDArray[np.int32]

  @staticmethod
  def create(file: pathlib.Path) -> Msh:
    """Create a Msh object from a .msh file."""

def msh_to_obj(msh_file: pathlib.Path) -> str:
  """Convert a legacy .msh file to the .obj format."""
