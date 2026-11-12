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
"""Interoperability between Warp and JAX (vendored copy)."""

from typing import Any, TypeAlias

import jax
import warp

_WarpScalar: TypeAlias = (
  type[warp.float16]
  | type[warp.bfloat16]
  | type[warp.float32]
  | type[warp.float64]
  | type[warp.int8]
  | type[warp.int16]
  | type[warp.int32]
  | type[warp.int64]
  | type[warp.uint8]
  | type[warp.uint16]
  | type[warp.uint32]
  | type[warp.uint64]
  | type[warp.bool]
)

def device_to_jax(warp_device: warp.DeviceLike) -> jax.Device:
  """Return the Jax device corresponding to a Warp device."""

def device_from_jax(jax_device: jax.Device) -> warp.Device:
  """Return the Warp device corresponding to a Jax device."""

def get_jax_device() -> jax.Device:
  """Get the current Jax device."""

def dtype_to_jax(warp_dtype: _WarpScalar) -> jax.typing.DTypeLike:
  """Return the Jax dtype corresponding to a Warp dtype."""

def dtype_from_jax(jax_dtype: jax.typing.DTypeLike) -> _WarpScalar:
  """Return the Warp dtype corresponding to a Jax dtype."""

def to_jax(warp_array: warp.array[Any]) -> jax.Array:
  """Convert a Warp array to a Jax array without copying the data."""

def from_jax(
  jax_array: jax.Array, dtype: type[Any] | None = ...
) -> warp.array[Any]:
  """Convert a Jax array to a Warp array without copying the data."""
