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
"""Built-in shapes for USD exporter."""

from typing import Any, Literal, TypeAlias, overload

import numpy as np
from numpy import typing as npt

import mujoco

_Real: TypeAlias = npt.NDArray[np.floating | np.integer]

@overload
def get_triangle_uvs(
  vertices: _Real, triangles: _Real, texture_type: None
) -> None: ...
@overload
def get_triangle_uvs(
  vertices: _Real, triangles: _Real, texture_type: mujoco.mjtTexture
) -> npt.NDArray[np.float64]: ...

class TriangleMesh:
  """Store UV and geometry information for a primitive mesh."""

  vertices: npt.NDArray[np.float64]
  triangles: npt.NDArray[np.int_]
  triangle_uvs: npt.NDArray[np.float64] | None
  def __init__(
    self,
    vertices: npt.NDArray[np.float64],
    triangles: npt.NDArray[np.int_],
    triangle_uvs: npt.NDArray[np.float64] | None,
  ) -> None: ...
  @classmethod
  def create_box(
    cls,
    width: float,
    height: float,
    depth: float,
    texture_type: mujoco.mjtTexture | None,
  ) -> TriangleMesh:
    """Creates a box."""

  @classmethod
  def create_sphere(
    cls,
    radius: float,
    texture_type: mujoco.mjtTexture | None,
    resolution: int,
  ) -> TriangleMesh:
    """Creates a sphere."""

  @classmethod
  def create_hemisphere(
    cls,
    radius: float,
    texture_type: mujoco.mjtTexture | None,
    resolution: int,
  ) -> TriangleMesh:
    """Creates a hemisphere."""

  @classmethod
  def create_cylinder(
    cls,
    radius: float,
    height: float,
    texture_type: mujoco.mjtTexture | None,
    resolution: int,
  ) -> TriangleMesh:
    """Creates a cylinder."""

  def translate(self, translation: npt.ArrayLike) -> None: ...
  def rotate(self, rotation: _Real, center: tuple[float, ...]) -> None: ...
  def scale(self, scale: npt.ArrayLike) -> None: ...
  def get_center(self) -> npt.NDArray[np.float64]: ...
  def __add__(self, other: TriangleMesh) -> TriangleMesh: ...

def decouple_config(config: dict[str, Any]) -> list[dict[str, Any]]:
  """Breaks a shape config into its subcomponent shapes."""

@overload
def mesh_config_generator(
  name: str,
  geom_type: int | mujoco.mjtGeom,
  size: _Real,
  decouple: Literal[False] = ...,
) -> dict[str, Any]: ...
@overload
def mesh_config_generator(
  name: str,
  geom_type: int | mujoco.mjtGeom,
  size: _Real,
  decouple: Literal[True],
) -> list[dict[str, Any]]: ...
@overload
def mesh_config_generator(
  name: str,
  geom_type: int | mujoco.mjtGeom,
  size: _Real,
  decouple: bool,
) -> dict[str, Any] | list[dict[str, Any]]: ...
def mesh_factory(
  mesh_config: dict[str, Any],
  texture_type: mujoco.mjtTexture | None,
  resolution: int = ...,
) -> tuple[str, TriangleMesh]:
  """Generates a mesh given a config consisting of shapes."""
