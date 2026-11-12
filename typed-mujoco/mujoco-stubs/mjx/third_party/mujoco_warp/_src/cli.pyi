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

"""Shared utilities and flags for MJWarp CLI tools."""

from collections.abc import Callable
from typing import TypeAlias

import numpy as np
from absl import flags
from etils import epath
from numpy import typing as npt

import mujoco
from mujoco.mjx.third_party.mujoco_warp._src.types import (
  Data,
  Model,
  RenderContext,
)
from mujoco.mjx.third_party.mujoco_warp._src.warp_util import _EventTrace

_Ctrls: TypeAlias = (
  list[npt.NDArray[np.float64]] | npt.NDArray[np.float32 | np.float64]
)

NWORLD: flags.FlagHolder[int]
NSTEP: flags.FlagHolder[int]
NCONMAX: flags.FlagHolder[int | None]
NJMAX: flags.FlagHolder[int | None]
NJMAX_NNZ: flags.FlagHolder[int | None]
NCCDMAX: flags.FlagHolder[int | None]
OVERRIDE: flags.FlagHolder[list[str]]
KEYFRAME: flags.FlagHolder[int]
EVENT_TRACE: flags.FlagHolder[bool]
NOISE_STD: flags.FlagHolder[float]
NOISE_RATE: flags.FlagHolder[float]
NVMAX: flags.FlagHolder[int | None]
INIT_ASLEEP: flags.FlagHolder[bool]
DEVICE: flags.FlagHolder[str | None]
REPLAY: flags.FlagHolder[str | None]
RENDER_WIDTH: flags.FlagHolder[list[int]]
RENDER_HEIGHT: flags.FlagHolder[list[int]]
RENDER_RGB: flags.FlagHolder[list[str]]
RENDER_DEPTH: flags.FlagHolder[list[str]]
RENDER_SEG: flags.FlagHolder[list[str]]
CAM_ACTIVE: flags.FlagHolder[list[str]]
CAMERA_NAME: flags.FlagHolder[list[str]]
RENDER_TEXTURES: flags.FlagHolder[bool]
RENDER_SHADOWS: flags.FlagHolder[bool]
RENDER_BACKFACE_CULLING: flags.FlagHolder[bool]
RENDER_SKYBOX: flags.FlagHolder[bool]

def load_model(path: epath.Path) -> mujoco.MjModel: ...
def init_structs(
  fn: Callable[..., None], mjm: mujoco.MjModel
) -> tuple[Model, Data, RenderContext | None, _Ctrls | None]: ...
def unroll(
  fn: Callable[..., None],
  m: Model,
  d: Data,
  rc: RenderContext | None,
  callback: Callable[[int, _EventTrace, float], None] | None = ...,
  ctrls: _Ctrls | None = ...,
) -> float: ...
