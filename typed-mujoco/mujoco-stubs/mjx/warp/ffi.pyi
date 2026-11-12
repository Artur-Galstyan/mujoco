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
"""FFI helper functions for MJX."""

import inspect
from collections.abc import Callable, Iterable
from typing import TypeVar

import jax
import warp as wp
from typing_extensions import Concatenate, ParamSpec

from mujoco.mjx._src import types

_P = ParamSpec('_P')
_R = TypeVar('_R')
_O = TypeVar('_O')
_B = TypeVar('_B')

def flatten_signature(
  signature: inspect.Signature, args: tuple[object, ...]
) -> inspect.Signature:
  """Flattens a tuple/dataclass signature."""

def jax_callable_variadic_tuple(
  func: Callable[..., None],
  num_outputs: int = ...,
  graph_mode: wp.JaxCallableGraphMode = ...,
  vmap_method: str | None = ...,
  output_dims: dict[str, tuple[int, ...]] | None = ...,
  in_out_argnames: Iterable[str] | None = ...,
  stage_in_argnames: Iterable[str] | None = ...,
  stage_out_argnames: Iterable[str] | None = ...,
  has_side_effect: bool = ...,
) -> Callable[..., list[jax.Array]]:
  """Wraps a JAX callable to support variadic tuples and dataclasses."""

def format_args_for_warp(
  func: Callable[_P, _R], verbose: bool = ...
) -> Callable[_P, _R]: ...
def marshal_jax_warp_callable(
  func: Callable[Concatenate[types.Model, types.Data, _P], _R],
  tree_map_output: bool = ...,
) -> Callable[Concatenate[types.Model, types.Data, _P], _R]:
  """Marshal fields into a MuJoCo Warp function."""

def marshal_custom_vmap(
  vmap_func: Callable[
    Concatenate[int, _B, types.Model, types.Data, _P], tuple[_R, _O]
  ],
  tree_map_output: bool = ...,
) -> Callable[Concatenate[int, _B, types.Model, types.Data, _P], tuple[_R, _O]]:
  """Marshal fields for a custom vmap into an MuJoCo Warp function."""
