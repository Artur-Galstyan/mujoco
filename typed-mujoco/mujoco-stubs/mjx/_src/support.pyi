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
"""Engine support functions."""

from collections.abc import Sequence

import jax
import numpy as np
from numpy import typing as npt

import mujoco
from mujoco.mjx._src.types import Data, Model

def is_sparse(m: mujoco.MjModel | Model) -> bool: ...
def make_m(
  m: Model, a: jax.Array, b: jax.Array, d: jax.Array | None = None
) -> jax.Array: ...
def full_m(m: Model, d: Data) -> jax.Array: ...
def mul_m(m: Model, d: Data, vec: jax.Array) -> jax.Array: ...
def jac(
  m: Model, d: Data, point: jax.Array, body_id: jax.Array
) -> tuple[jax.Array, jax.Array]: ...
def jac_dot(
  m: Model, d: Data, point: jax.Array, body_id: jax.Array
) -> tuple[jax.Array, jax.Array]: ...
def apply_ft(
  m: Model,
  d: Data,
  force: jax.Array,
  torque: jax.Array,
  point: jax.Array,
  body_id: jax.Array,
) -> jax.Array: ...
def xfrc_accumulate(m: Model, d: Data) -> jax.Array: ...
def local_to_global(
  world_pos: jax.Array,
  world_quat: jax.Array,
  local_pos: jax.Array,
  local_quat: jax.Array,
) -> tuple[jax.Array, jax.Array]: ...
def id2name(
  m: Model | mujoco.MjModel, typ: mujoco.mjtObj, i: int
) -> str | None: ...
def name2id(
  m: Model | mujoco.MjModel, typ: mujoco.mjtObj, name: str
) -> int: ...

class BindModel(object):
  model: Model
  prefix: str
  id: int | list[int]
  def __init__(self, model: Model, specs: Sequence[mujoco.MjStruct]): ...
  def __getattr__(self, name: str) -> jax.Array: ...

class BindData(object):
  data: Data
  model: Model
  prefix: str
  id: int | list[int]
  def __init__(
    self, data: Data, model: Model, specs: Sequence[mujoco.MjStruct]
  ): ...
  def __getattr__(self, name: str) -> jax.Array: ...
  def set(self, name: str, value: jax.Array) -> Data: ...

def contact_force(
  m: Model, d: Data, contact_id: int, to_world_frame: bool = False
) -> jax.Array: ...
def contact_force_dim(
  m: Model, d: Data, dim: int
) -> tuple[jax.Array, npt.NDArray[np.intp]]: ...
def wrap_circle(
  d: jax.Array, sd: jax.Array, sidesite: jax.Array, rad: jax.Array
) -> tuple[jax.Array, jax.Array]: ...
def wrap_inside(
  end: jax.Array,
  radius: jax.Array,
  maxiter: int,
  tolerance: float,
  z_init: float,
) -> tuple[jax.Array, jax.Array]: ...
def wrap(
  x0: jax.Array,
  x1: jax.Array,
  xpos: jax.Array,
  xmat: jax.Array,
  size: jax.Array,
  side: jax.Array,
  sidesite: jax.Array,
  is_sphere: jax.Array,
  is_wrap_inside: bool,
  wrap_inside_maxiter: int,
  wrap_inside_tolerance: float,
  wrap_inside_z_init: float,
) -> tuple[jax.Array, jax.Array, jax.Array]: ...
def muscle_gain_length(
  length: jax.Array, lmin: jax.Array, lmax: jax.Array
) -> jax.Array: ...
def muscle_gain(
  length: jax.Array,
  vel: jax.Array,
  lengthrange: jax.Array,
  acc0: jax.Array,
  prm: jax.Array,
) -> jax.Array: ...
def muscle_bias(
  length: jax.Array, lengthrange: jax.Array, acc0: jax.Array, prm: jax.Array
) -> jax.Array: ...
def muscle_dynamics_timescale(
  dctrl: jax.Array,
  tau_act: jax.Array,
  tau_deact: jax.Array,
  smoothing_width: jax.Array,
) -> jax.Array: ...
def muscle_dynamics(
  ctrl: jax.Array, act: jax.Array, prm: jax.Array
) -> jax.Array: ...
