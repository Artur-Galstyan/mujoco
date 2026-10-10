# Copyright 2026 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS.

from collections.abc import Sequence

import numpy as np
from numpy import typing as npt

from mujoco import MjModel, MjvCamera, MjvGeom, MjvOption, MjvPerturb

MAX_EXTRA_GEOMS: int

def serialize_state_payload(
  model_crc32: int,
  physics_spec: int,
  physics_state: bytes,
  camera: MjvCamera,
  perturb: MjvPerturb,
  vis_options: MjvOption,
  model: MjModel,
  render_flags: Sequence[int] | npt.NDArray[np.uint8],
  extra_geoms: Sequence[MjvGeom],
  /,
) -> bytes:
  """Serialize physics state and render state for the browser's state stream."""

def max_state_payload_size(physics_bytes: int, /) -> int:
  """Get the payload capacity required for physics state of this byte size."""
