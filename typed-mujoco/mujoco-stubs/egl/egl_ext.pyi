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

"""Extends OpenGL.EGL with definitions necessary for headless rendering."""

import ctypes
from collections.abc import Callable

from OpenGL.EGL import *  # noqa: F403

class _OpaquePointer(ctypes._Pointer[ctypes.Structure]):
  @property
  def address(self) -> int: ...
  @property
  def as_voidp(self) -> ctypes.c_void_p: ...
  def __hash__(self) -> int: ...

class _EGLDisplay(_OpaquePointer): ...
class _EGLDeviceEXT(_OpaquePointer): ...

EGLDisplay = _EGLDisplay
EGLDeviceEXT = _EGLDeviceEXT

EGL_NO_DISPLAY: _EGLDisplay
EGL_CUDA_DEVICE_NV: int
PFNEGLQUERYDEVICESEXTPROC: type[ctypes._CFunctionType]
EGL_PLATFORM_DEVICE_EXT: int
PFNEGLGETPLATFORMDISPLAYEXTPROC: type[ctypes._CFunctionType]
PFNEGLQUERYDEVICEATTRIBEXTPROC: type[ctypes._CFunctionType]
eglGetPlatformDisplayEXT: Callable[
  [
    int,
    _EGLDeviceEXT | ctypes.c_void_p | None,
    ctypes._Pointer[ctypes.c_int32] | None,
  ],
  _EGLDisplay,
]

def eglQueryDevicesEXT(
  max_devices: int = 10,
) -> list[_EGLDeviceEXT]: ...
def eglQueryDeviceAttribEXT(device: _EGLDeviceEXT, attribute: int) -> int: ...
