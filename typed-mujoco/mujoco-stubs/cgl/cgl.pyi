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

"""Bindings for Apple CGL."""

import ctypes
import enum
from typing import TypeAlias

CGLContextObj: TypeAlias = ctypes.c_void_p
CGLPixelFormatObj: TypeAlias = ctypes.c_void_p
GLint: TypeAlias = ctypes.c_int

class CGLOpenGLProfile(enum.IntEnum):
  CGLOGLPVersion_Legacy = 0x1000
  CGLOGLPVersion_3_2_Core = 0x3200
  CGLOGLPVersion_GL3_Core = 0x3200
  CGLOGLPVersion_GL4_Core = 0x4100

class CGLPixelFormatAttribute(enum.IntEnum):
  """CGLPixelFormatAttribute enum values."""

  CGLPFAAllRenderers = 1
  CGLPFATripleBuffer = 3
  CGLPFADoubleBuffer = 5
  CGLPFAColorSize = 8
  CGLPFAAlphaSize = 11
  CGLPFADepthSize = 12
  CGLPFAStencilSize = 13
  CGLPFAMinimumPolicy = 51
  CGLPFAMaximumPolicy = 52
  CGLPFASampleBuffers = 55
  CGLPFASample = 56
  CGLPFAColorFloat = 58
  CGLPFAMultisample = 59
  CGLPFASupersample = 60
  CGLPFASampleAlpha = 61
  CGLPFARendererID = 70
  CGLPFANoRecovery = 72
  CGLPFAAccelerated = 73
  CGLPFAClosestPolicy = 74
  CGLPFABackingStore = 76
  CGLPFABackingVolatile = 77
  CGLPFADisplayMask = 84
  CGLPFAAllowOfflineRenderers = 96
  CGLPFAAcceleratedCompute = 97
  CGLPFAOpenGLProfile = 99
  CGLPFASupportsAutomaticGraphicsSwitching = 101
  CGLPFAVirtualScreenCount = 128
  CGLPFAAuxBuffers = 7
  CGLPFAAccumSize = 14
  CGLPFAAuxDepthStencil = 57
  CGLPFAStereo = 6
  CGLPFAOffScreen = 53
  CGLPFAWindow = 80
  CGLPFACompliant = 83
  CGLPFAPBuffer = 90
  CGLPFARemotePBuffer = 91
  CGLPFASingleRenderer = 71
  CGLPFARobust = 75
  CGLPFAMPSafe = 78
  CGLPFAMultiScreen = 81
  CGLPFAFullScreen = 54

class CGLError(RuntimeError): ...

_CGLArg: TypeAlias = ctypes._CData | ctypes._CArgObject | int | None

def CGLChoosePixelFormat(*args: _CGLArg) -> None: ...
def CGLCreateContext(*args: _CGLArg) -> None: ...
def CGLLockContext(*args: _CGLArg) -> None: ...
def CGLSetCurrentContext(*args: _CGLArg) -> None: ...
def CGLUnlockContext(*args: _CGLArg) -> None: ...

CGLReleaseContext: ctypes._NamedFuncPointer
CGLReleasePixelFormat: ctypes._NamedFuncPointer
