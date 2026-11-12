# Copyright 2026 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS.

from collections.abc import Iterable, Sequence

import numpy as np
from numpy import typing as npt

from mujoco import MjModel, MjvCamera, MjvGeom, MjvOption, MjvPerturb

MAX_EXTRA_GEOMS: int

class ClientCameraMessage:
  @property
  def generation(self) -> int: ...
  @property
  def type(self) -> int: ...
  @property
  def fixedcamid(self) -> int: ...
  @property
  def trackbodyid(self) -> int: ...
  @property
  def orthographic(self) -> int: ...
  @property
  def lookat(self) -> tuple[float, float, float]: ...
  @property
  def distance(self) -> float: ...
  @property
  def azimuth(self) -> float: ...
  @property
  def elevation(self) -> float: ...

class ViewerInputState:
  def __init__(
    self,
    *,
    mouse_focus: bool = False,
    camera_index: int = 0,
    keyboard_focus: bool = False,
    camera_generation: int = 0,
  ) -> None: ...
  mouse_focus: int
  camera_index: int
  keyboard_focus: int
  camera_generation: int

def parse_client_camera(message: bytes) -> ClientCameraMessage | None: ...
def serialize_client_camera(camera: MjvCamera, generation: int) -> bytes: ...
def parse_model_bundle(
  bundle: bytes, /
) -> tuple[bytes, list[tuple[str, bytes]]]:
  """Split a model bundle into the MJB bytes and the (path, bytes) texture files."""

def serialize_model_bundle(model: MjModel, /) -> bytes:
  """Serialize the model and the files of its textures for the browser."""

def serialize_state_payload(
  model_crc32: int,
  physics_spec: int,
  physics_state: bytes,
  camera: MjvCamera,
  perturb: MjvPerturb,
  vis_options: MjvOption,
  model: MjModel,
  render_flags: Sequence[int] | npt.NDArray[np.uint8],
  extra_geoms: Iterable[MjvGeom],
  viewer_input: ViewerInputState,
) -> bytes:
  """Serialize physics state and render state for the browser's state stream."""

def max_state_payload_size(physics_bytes: int, /) -> int:
  """Get the payload capacity required for physics state of this byte size."""
