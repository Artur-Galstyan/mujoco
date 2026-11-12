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
"""ctypes structures and helpers for XLA's FFI API."""

import ctypes
import enum
from typing import TypeAlias

import jax
import numpy as np
import warp as wp
from numpy import typing as npt

_WarpScalar: TypeAlias = (
  type[wp.int8]
  | type[wp.int16]
  | type[wp.int32]
  | type[wp.int64]
  | type[wp.uint8]
  | type[wp.uint16]
  | type[wp.uint32]
  | type[wp.uint64]
  | type[wp.float16]
  | type[wp.float32]
  | type[wp.float64]
)
_Attr: TypeAlias = str | npt.NDArray[np.generic] | dict[str, _Attr]

class XLA_FFI_Extension_Type(enum.IntEnum):
  Metadata = 1

class XLA_FFI_Extension_Base(ctypes.Structure):
  struct_size: int
  type: int
  next: ctypes._Pointer[XLA_FFI_Extension_Base]

class XLA_FFI_ExecutionStage(enum.IntEnum):
  INSTANTIATE = 0
  PREPARE = 1
  INITIALIZE = 2
  EXECUTE = 3

class XLA_FFI_DataType(enum.IntEnum):
  INVALID = 0
  PRED = 1
  S8 = 2
  S16 = 3
  S32 = 4
  S64 = 5
  U8 = 6
  U16 = 7
  U32 = 8
  U64 = 9
  F16 = 10
  F32 = 11
  F64 = 12
  BF16 = 16
  C64 = 15
  C128 = 18
  TOKEN = 17
  F8E5M2 = 19
  F8E3M4 = 29
  F8E4M3 = 28
  F8E4M3FN = 20
  F8E4M3B11FNUZ = 23
  F8E5M2FNUZ = 24
  F8E4M3FNUZ = 25
  F4E2M1FN = 32
  F8E8M0FNU = 33

class XLA_FFI_Buffer(ctypes.Structure):
  struct_size: int
  extension_start: ctypes._Pointer[XLA_FFI_Extension_Base]
  dtype: int
  data: int | None
  rank: int
  dims: ctypes._Pointer[ctypes.c_int64]

class XLA_FFI_ArgType(enum.IntEnum):
  BUFFER = 1

class XLA_FFI_RetType(enum.IntEnum):
  BUFFER = 1

class XLA_FFI_Args(ctypes.Structure):
  struct_size: int
  extension_start: ctypes._Pointer[XLA_FFI_Extension_Base]
  size: int
  types: ctypes._Pointer[ctypes.c_int]
  args: ctypes._Pointer[ctypes.c_void_p]

class XLA_FFI_Rets(ctypes.Structure):
  struct_size: int
  extension_start: ctypes._Pointer[XLA_FFI_Extension_Base]
  size: int
  types: ctypes._Pointer[ctypes.c_int]
  rets: ctypes._Pointer[ctypes.c_void_p]

class XLA_FFI_ByteSpan(ctypes.Structure):
  ptr: ctypes._Pointer[ctypes.c_char]
  len: int

class XLA_FFI_Scalar(ctypes.Structure):
  dtype: int
  value: int | None

class XLA_FFI_Array(ctypes.Structure):
  dtype: int
  size: int
  data: int | None

class XLA_FFI_AttrType(enum.IntEnum):
  ARRAY = 1
  DICTIONARY = 2
  SCALAR = 3
  STRING = 4

class XLA_FFI_Attrs(ctypes.Structure):
  struct_size: int
  extension_start: ctypes._Pointer[XLA_FFI_Extension_Base]
  size: int
  types: ctypes._Pointer[ctypes.c_int]
  names: ctypes._Pointer[ctypes._Pointer[XLA_FFI_ByteSpan]]
  attrs: ctypes._Pointer[ctypes.c_void_p]

class XLA_FFI_Api_Version(ctypes.Structure):
  struct_size: int
  extension_start: ctypes._Pointer[XLA_FFI_Extension_Base]
  major_version: int
  minor_version: int

class XLA_FFI_Handler_TraitsBits(enum.IntEnum):
  COMMAND_BUFFER_COMPATIBLE = 1

class XLA_FFI_Metadata(ctypes.Structure):
  struct_size: int
  api_version: XLA_FFI_Api_Version
  traits: int

class XLA_FFI_Metadata_Extension(ctypes.Structure):
  extension_base: XLA_FFI_Extension_Base
  metadata: ctypes._Pointer[XLA_FFI_Metadata]

class XLA_FFI_Error_Code(enum.IntEnum):
  OK = 0
  CANCELLED = 1
  UNKNOWN = 2
  INVALID_ARGUMENT = 3
  DEADLINE_EXCEEDED = 4
  NOT_FOUND = 5
  ALREADY_EXISTS = 6
  PERMISSION_DENIED = 7
  RESOURCE_EXHAUSTED = 8
  FAILED_PRECONDITION = 9
  ABORTED = 10
  OUT_OF_RANGE = 11
  UNIMPLEMENTED = 12
  INTERNAL = 13
  UNAVAILABLE = 14
  DATA_LOSS = 15
  UNAUTHENTICATED = 16

class XLA_FFI_Error_Create_Args(ctypes.Structure):
  struct_size: int
  extension_start: ctypes._Pointer[XLA_FFI_Extension_Base]
  message: bytes | None
  errc: int

XLA_FFI_Error_Create: type[ctypes._CFunctionType]

class XLA_FFI_Stream_Get_Args(ctypes.Structure):
  struct_size: int
  extension_start: ctypes._Pointer[XLA_FFI_Extension_Base]
  ctx: int | None
  stream: int | None

XLA_FFI_Stream_Get: type[ctypes._CFunctionType]

class XLA_FFI_DeviceOrdinal_Get_Args(ctypes.Structure):
  struct_size: int
  extension_start: ctypes._Pointer[XLA_FFI_Extension_Base]
  ctx: int | None
  device_ordinal: int

XLA_FFI_DeviceOrdinal_Get: type[ctypes._CFunctionType]

class XLA_FFI_Api(ctypes.Structure):
  struct_size: int
  extension_start: ctypes._Pointer[XLA_FFI_Extension_Base]
  api_version: XLA_FFI_Api_Version
  internal_api: int | None
  XLA_FFI_Error_Create: ctypes._CFunctionType
  XLA_FFI_Error_GetMessage: int | None
  XLA_FFI_Error_Destroy: int | None
  XLA_FFI_Handler_Register: int | None
  XLA_FFI_Stream_Get: ctypes._CFunctionType
  XLA_FFI_TypeId_Register: int | None
  XLA_FFI_ExecutionContext_Get: int | None
  XLA_FFI_State_Set: int | None
  XLA_FFI_State_Get: int | None
  XLA_FFI_DeviceMemory_Allocate: int | None
  XLA_FFI_DeviceMemory_Free: int | None
  XLA_FFI_ThreadPool_Schedule: int | None
  XLA_FFI_ThreadPool_NumThreads: int | None
  XLA_FFI_Future_Create: int | None
  XLA_FFI_Future_SetAvailable: int | None
  XLA_FFI_Future_SetError: int | None
  XLA_FFI_RunId_Get: int | None
  XLA_FFI_DeviceOrdinal_Get: ctypes._CFunctionType

class XLA_FFI_CallFrame(ctypes.Structure):
  struct_size: int
  extension_start: ctypes._Pointer[XLA_FFI_Extension_Base]
  api: ctypes._Pointer[XLA_FFI_Api]
  ctx: int | None
  stage: int
  args: XLA_FFI_Args
  rets: XLA_FFI_Rets
  attrs: XLA_FFI_Attrs
  future: int | None

def decode_bytespan(span: XLA_FFI_ByteSpan) -> str: ...
def decode_scalar(scalar: XLA_FFI_Scalar) -> npt.NDArray[np.generic]: ...
def decode_array(array: XLA_FFI_Array) -> npt.NDArray[np.generic]: ...
def decode_attrs(attrs: XLA_FFI_Attrs) -> dict[str, _Attr]: ...
def create_ffi_error(
  api: ctypes._Pointer[XLA_FFI_Api],
  errc: XLA_FFI_Error_Code | int,
  message: str,
) -> int | None: ...
def create_invalid_argument_ffi_error(
  api: ctypes._Pointer[XLA_FFI_Api], message: str
) -> int | None: ...
def get_stream_from_callframe(call_frame: XLA_FFI_CallFrame) -> int | None: ...
def get_device_ordinal_from_callframe(call_frame: XLA_FFI_CallFrame) -> int: ...
def dtype_from_ffi(ffi_dtype: XLA_FFI_DataType | int) -> _WarpScalar | None: ...
def jax_dtype_from_ffi(
  ffi_dtype: XLA_FFI_DataType | int,
) -> jax.typing.DTypeLike | None: ...

class ExecutionContext:
  stage: XLA_FFI_ExecutionStage
  stream: int
  def __init__(self, callframe: XLA_FFI_CallFrame) -> None: ...

class FfiBuffer:
  dtype: np.dtype[np.generic]
  data: int | None
  shape: tuple[int, ...]
  def __init__(self, xla_buffer: XLA_FFI_Buffer) -> None: ...
  @property
  def __cuda_array_interface__(
    self,
  ) -> dict[str, tuple[int, ...] | str | tuple[int | None, bool] | int]: ...
