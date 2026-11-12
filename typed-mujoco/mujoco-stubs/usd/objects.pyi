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
"""Objects module for USD exporter."""

import abc
import collections
from collections.abc import Sequence
from typing import Any, TypeAlias, TypedDict

import numpy as np
from numpy import typing as npt

import mujoco
from mujoco.usd.shapes import TriangleMesh

_Stage: TypeAlias = Any
_Prim: TypeAlias = Any
_Xform: TypeAlias = Any
_XformOp: TypeAlias = Any
_Mesh: TypeAlias = Any
_Primvar: TypeAlias = Any
_Real: TypeAlias = npt.NDArray[np.floating | np.integer]
_GeomTextures: TypeAlias = Sequence[tuple[str, mujoco.mjtTexture] | None]

class _TendonPartRefs(TypedDict, total=False):
  usd_xform: _Xform
  usd_mesh: _Mesh
  usd_prim: _Prim
  translate_op: _XformOp
  scale_op: _XformOp

class USDObject(abc.ABC):
  """Abstract interface for all USD objects including meshes and primitives."""

  stage: _Stage
  model: mujoco.MjModel
  geom: mujoco.MjvGeom
  obj_name: str
  rgba: _Real
  geom_textures: _GeomTextures
  xform_path: str
  usd_xform: _Xform
  transform_op: _XformOp
  scale_op: _XformOp
  last_visible_frame: int
  def __init__(
    self,
    stage: _Stage,
    model: mujoco.MjModel,
    geom: mujoco.MjvGeom,
    obj_name: str,
    rgba: _Real = ...,
    geom_textures: _GeomTextures = ...,
  ) -> None: ...
  def attach_image_material(self, usd_mesh: _Mesh) -> None:
    """Attaches an image texture to a material for a USD object."""

  def attach_solid_material(self, usd_mesh: _Mesh) -> None:
    """Attaches an solid texture to a material for a USD object."""

  def update(
    self,
    pos: _Real,
    mat: _Real,
    visible: bool,
    frame: int,
    scale: _Real | None = ...,
  ) -> None:
    """Updates the position and orientation of an object."""

  def update_visibility(self, visible: bool, frame: int) -> None:
    """Updates the visibility of an object in a scene for a given frame."""

  def update_scale(self, scale: _Real, frame: int) -> None:
    """Updates the scale of an object in the scene for a given frame."""

class USDMesh(USDObject):
  """Class that handles predefined meshes in the USD scene."""

  dataid: int
  usd_mesh: _Mesh
  usd_prim: _Prim
  texcoords: _Primvar
  def __init__(
    self,
    stage: _Stage,
    model: mujoco.MjModel,
    geom: mujoco.MjvGeom,
    obj_name: str,
    dataid: int,
    rgba: _Real = ...,
    geom_textures: _GeomTextures = ...,
  ) -> None: ...

class USDPrimitiveMesh(USDObject):
  """Class to handle primitive shapes in the USD scene."""

  mesh_config: dict[str, Any]
  prim_mesh: TriangleMesh
  usd_mesh: _Mesh
  usd_prim: _Prim
  texcoords: _Primvar
  def __init__(
    self,
    mesh_config: dict[str, Any],
    stage: _Stage,
    model: mujoco.MjModel,
    geom: mujoco.MjvGeom,
    obj_name: str,
    rgba: _Real = ...,
    geom_textures: _GeomTextures = ...,
  ) -> None: ...
  def generate_primitive_mesh(self) -> TriangleMesh:
    """Generates the mesh for the primitive USD object."""

class USDTendon(USDObject):
  """Class to handle tendons in the USD scene."""

  mesh_config: list[dict[str, Any]]
  tendon_parts: dict[str, TriangleMesh]
  usd_refs: collections.defaultdict[str, _TendonPartRefs]
  texcoords: _Primvar
  def __init__(
    self,
    mesh_config: list[dict[str, Any]],
    stage: _Stage,
    model: mujoco.MjModel,
    geom: mujoco.MjvGeom,
    obj_name: str,
    rgba: _Real = ...,
    geom_textures: _GeomTextures = ...,
  ) -> None: ...
  def generate_primitive_mesh(self) -> dict[str, TriangleMesh]:
    """Generates the tendon mesh using primitives."""

  def update(
    self,
    pos: _Real,
    mat: _Real,
    visible: bool,
    frame: int,
    scale: _Real | None = ...,
  ) -> None:
    """Updates the position and orientation of an object in the scene."""

  def update_scale(self, scale: _Real, frame: int) -> None:
    """Updates the scale of the tendon."""
