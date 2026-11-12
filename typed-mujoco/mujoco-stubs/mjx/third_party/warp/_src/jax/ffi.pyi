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
"""JAX FFI callables for Warp kernels and functions (vendored copy)."""

import collections
import ctypes
from collections.abc import Callable, Iterable, Sequence
from enum import IntEnum
from typing import Any, Literal, Protocol, TypeAlias, overload

import jax
import warp as wp

from mujoco.mjx.third_party.warp._src.jax.xla_ffi import *  # noqa: F403
from mujoco.mjx.third_party.warp._src.jax.xla_ffi import (
  XLA_FFI_CallFrame,
)

_Dims: TypeAlias = int | Sequence[int]
_OutputDims: TypeAlias = _Dims | dict[str, _Dims] | None
_ArgType: TypeAlias = wp.array[Any] | type[Any]

JAX_CALLABLE_DEFAULT_GRAPH_CACHE_MAX: int
ffi_name_counts: dict[str, int]

def check_jax_version() -> None: ...
def collapse_batch_dims(
  shape: tuple[int, ...], desired_ndim: int
) -> tuple[int, ...]: ...
def compute_batch_size(shape: Sequence[int], batch_ndim: int) -> int: ...

class JaxCallableGraphMode(IntEnum):
  """CUDA graph capture modes for :func:`warp.jax_callable`."""

  NONE = 0
  JAX = 1
  WARP = 2
  WARP_STAGED = 3
  WARP_STAGED_EX = 4

GraphMode = JaxCallableGraphMode

class JaxModulePreloadMode(IntEnum):
  """Module preload modes for JAX interop callables."""

  NONE = 0
  CURRENT_DEVICE = 1
  ALL_DEVICES = 2

ModulePreloadMode = JaxModulePreloadMode

class FfiArg:
  name: str
  type: _ArgType
  in_out: bool
  is_array: bool
  dtype_shape: tuple[int, ...]
  dtype_ndim: int
  jax_scalar_type: jax.typing.DTypeLike
  jax_ndim: int
  warp_ndim: int
  def __init__(self, name: str, type: _ArgType, in_out: bool = ...) -> None: ...

class FfiLaunchDesc:
  static_inputs: dict[str, object]
  launch_dims: tuple[int, ...] | None
  def __init__(
    self,
    static_inputs: dict[str, object],
    launch_dims: tuple[int, ...] | None,
  ) -> None: ...

class FfiKernel:
  kernel: wp.Kernel
  name: str
  num_outputs: int
  vmap_method: str | None
  launch_dims: _Dims | None
  block_dim: int | None
  output_dims: _OutputDims
  module_preload_mode: JaxModulePreloadMode
  has_side_effect: bool
  first_array_arg: int | None
  launch_id: int
  launch_descriptors: dict[int, FfiLaunchDesc]
  num_kernel_args: int
  num_in_out: int
  num_inputs: int
  input_args: list[FfiArg]
  output_args: list[FfiArg]
  input_output_aliases: dict[int, int]
  callback_funcs: dict[str, ctypes._CFunctionType]
  def __init__(
    self,
    kernel: wp.Kernel,
    num_outputs: int,
    vmap_method: str | None,
    launch_dims: _Dims | None,
    block_dim: int | None,
    output_dims: _OutputDims,
    in_out_argnames: Iterable[str] | None,
    module_preload_mode: JaxModulePreloadMode,
    has_side_effect: bool = ...,
  ) -> None: ...
  def __call__(
    self,
    *args: jax.typing.ArrayLike,
    output_dims: _OutputDims = ...,
    launch_dims: _Dims | None = ...,
    vmap_method: str | None = ...,
  ) -> list[jax.Array]: ...
  def ffi_callback(
    self, call_frame: ctypes._Pointer[XLA_FFI_CallFrame], platform: str
  ) -> int | None: ...

class FfiCallDesc:
  static_inputs: dict[str, object]
  capture: wp.ScopedCapture | None
  input_staging_arrays: list[wp.array[Any]] | None
  output_staging_arrays: list[wp.array[Any]] | None
  static_staging_arrays: list[wp.array[Any]] | None
  input_memcpy_count: int
  input_memcpy_indices: list[int] | None
  input_memcpy_srcs: ctypes.Array[ctypes.c_void_p] | None
  input_memcpy_dsts: ctypes.Array[ctypes.c_void_p] | None
  input_memcpy_sizes: ctypes.Array[ctypes.c_size_t] | None
  input_memcpy_kinds: ctypes.Array[ctypes.c_int] | None
  output_memcpy_count: int
  output_memcpy_indices: list[int] | None
  output_memcpy_srcs: ctypes.Array[ctypes.c_void_p] | None
  output_memcpy_dsts: ctypes.Array[ctypes.c_void_p] | None
  output_memcpy_sizes: ctypes.Array[ctypes.c_size_t] | None
  output_memcpy_kinds: ctypes.Array[ctypes.c_int] | None
  memcpy_nodes: ctypes.Array[ctypes.c_void_p]
  memcpy_srcs: ctypes.Array[ctypes.c_void_p]
  memcpy_dsts: ctypes.Array[ctypes.c_void_p]
  memcpy_sizes: ctypes.Array[ctypes.c_size_t]
  memcpy_kinds: ctypes.Array[ctypes.c_int]
  def __init__(self, static_inputs: dict[str, object]) -> None: ...

class FfiCallable:
  func: Callable[..., None]
  name: str
  num_outputs: int
  vmap_method: str | None
  graph_mode: JaxCallableGraphMode
  output_dims: _OutputDims
  module_preload_mode: JaxModulePreloadMode
  has_side_effect: bool
  first_array_arg: int | None
  call_id: int
  call_descriptors: dict[int, FfiCallDesc]
  captures: collections.OrderedDict[int, wp.ScopedCapture]
  stage_in_argnames: set[str] | None
  stage_out_argnames: set[str] | None
  num_in_out: int
  num_inputs: int
  args: list[FfiArg]
  arg_input_indices: list[int | None]
  arg_output_indices: list[int | None]
  input_args: list[FfiArg]
  output_args: list[FfiArg]
  array_input_indices: list[int]
  array_output_indices: list[int]
  input_output_aliases: dict[int, int]
  callback_funcs: dict[str, ctypes._CFunctionType]
  def __init__(
    self,
    func: Callable[..., None],
    num_outputs: int,
    graph_mode: JaxCallableGraphMode,
    vmap_method: str | None,
    output_dims: _OutputDims,
    in_out_argnames: Iterable[str] | None,
    stage_in_argnames: Iterable[str] | None,
    stage_out_argnames: Iterable[str] | None,
    graph_cache_max: int | None,
    module_preload_mode: JaxModulePreloadMode,
    has_side_effect: bool = ...,
  ) -> None: ...
  def __call__(
    self,
    *args: jax.typing.ArrayLike,
    output_dims: _OutputDims = ...,
    vmap_method: str | None = ...,
  ) -> list[jax.Array]: ...
  def ffi_callback(
    self, call_frame: ctypes._Pointer[XLA_FFI_CallFrame], platform: str
  ) -> int | None: ...
  @property
  def graph_cache_max(self) -> int | None: ...
  @graph_cache_max.setter
  def graph_cache_max(self, value: int | None) -> None: ...
  @property
  def graph_cache_size(self) -> int: ...

class _DifferentiableKernel(Protocol):
  block_dim: int | None
  def __call__(
    self,
    *args: jax.typing.ArrayLike,
    launch_dims: object = ...,
    output_dims: object = ...,
    vmap_method: object = ...,
  ) -> list[jax.Array]: ...

@overload
def jax_kernel(
  kernel: wp.Kernel,
  num_outputs: int = ...,
  vmap_method: str | None = ...,
  launch_dims: _Dims | None = ...,
  output_dims: _OutputDims = ...,
  in_out_argnames: Iterable[str] | None = ...,
  module_preload_mode: JaxModulePreloadMode = ...,
  enable_backward: Literal[False] = ...,
  has_side_effect: bool = ...,
  block_dim: int | None = ...,
) -> FfiKernel: ...
@overload
def jax_kernel(
  kernel: wp.Kernel,
  num_outputs: int = ...,
  vmap_method: str | None = ...,
  launch_dims: _Dims | None = ...,
  output_dims: None = ...,
  in_out_argnames: None = ...,
  module_preload_mode: JaxModulePreloadMode = ...,
  *,
  enable_backward: Literal[True],
  has_side_effect: bool = ...,
  block_dim: int | None = ...,
) -> _DifferentiableKernel: ...
def jax_callable(
  func: Callable[..., None],
  num_outputs: int = ...,
  graph_mode: JaxCallableGraphMode = ...,
  vmap_method: str | None = ...,
  output_dims: _OutputDims = ...,
  in_out_argnames: Iterable[str] | None = ...,
  stage_in_argnames: Iterable[str] | None = ...,
  stage_out_argnames: Iterable[str] | None = ...,
  graph_cache_max: int | None = ...,
  module_preload_mode: JaxModulePreloadMode = ...,
  has_side_effect: bool = ...,
) -> FfiCallable:
  """Create a JAX callback from an annotated Python function."""

def clear_jax_callable_graph_cache(callable: FfiCallable | None = ...) -> None:
  """Clear the graph cache of the given callable or all callables."""

def register_ffi_callback(
  name: str, func: Callable[..., None], graph_compatible: bool = ...
) -> None:
  """Create a JAX callback from a Python function."""

def generate_unique_name(func: Callable[..., object]) -> str: ...
def get_warp_shape(arg: FfiArg, dims: Sequence[int]) -> Sequence[int]: ...
def get_jax_output_type_metadata(
  input_values: Iterable[Any],
) -> dict[str, Any]: ...
def get_jax_output_type(
  arg: FfiArg, dims: _Dims, metadata: dict[str, Any] | None = ...
) -> jax.ShapeDtypeStruct: ...
