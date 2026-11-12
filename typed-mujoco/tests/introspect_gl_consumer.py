# Copyright 2026 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS.
"""Static introspect and GL context consumer; ty checks it, never run it."""

import contextlib
import ctypes
import dataclasses
import re
from collections.abc import Mapping
from typing import Any, Literal

from typing_extensions import assert_type

import mujoco
from mujoco import _version, cgl, egl, glfw, osmesa, rendering
from mujoco.cgl import cgl as cgl_bindings
from mujoco.egl import egl_ext
from mujoco.introspect import (
  ast_nodes,
  enums,
  functions,
  mjxmacro,
  structs,
  type_parsing,
)
from mujoco.introspect.codegen import ast_processor, formatter, generate
from mujoco.rendering import classic
from mujoco.rendering.classic import gl_context, renderer


def version_info() -> None:
  assert_type(_version.__version__, str)
  assert_type(mujoco.__version__, str)
  assert rendering is not None
  assert classic is not None


def ast_types() -> None:
  value = ast_nodes.ValueType('int', is_const=True)
  assert_type(value.name, str)
  assert_type(value.is_const, bool)
  assert_type(value.decl('x'), str)
  array = ast_nodes.ArrayType(value, [3, 4])
  assert_type(array.extents, tuple[int, ...])
  assert_type(array.inner_type, ast_nodes.ValueType | ast_nodes.PointerType)
  pointer = ast_nodes.PointerType(array, nullable=True)
  assert_type(
    pointer.inner_type,
    ast_nodes.ValueType | ast_nodes.ArrayType | ast_nodes.PointerType,
  )
  param = ast_nodes.FunctionParameterDecl('x', pointer)
  assert_type(param.decltype, str)
  decl = ast_nodes.FunctionDecl('f', value, [param], 'doc')
  assert_type(decl.parameters, tuple[ast_nodes.FunctionParameterDecl, ...])
  assert_type(decl.decltype, str)
  enum = ast_nodes.EnumDecl('e', 'enum e', {'A': 1})
  assert_type(enum.values, dict[str, int])
  field = ast_nodes.StructFieldDecl('a', value, 'doc', (1, 'n'))
  assert_type(field.array_extent, tuple[str | int, ...] | None)
  union = ast_nodes.AnonymousUnionDecl([field])
  assert_type(
    union.fields,
    tuple[ast_nodes.StructFieldDecl | ast_nodes.AnonymousUnionDecl, ...],
  )
  struct = ast_nodes.StructDecl('s', 'struct s', [field, union])
  assert_type(struct.decl(), str)
  assert dataclasses.is_dataclass(struct)
  assert_type(ast_nodes.VALID_TYPE_NAME_PATTERN, re.Pattern[str])
  assert_type(ast_nodes.C_INVALID_TYPE_NAMES, frozenset[str])


def tables() -> None:
  assert_type(enums.ENUMS, Mapping[str, ast_nodes.EnumDecl])
  assert_type(functions.FUNCTIONS['mj_step'], ast_nodes.FunctionDecl)
  assert_type(
    structs.STRUCTS['mjModel'].fields,
    tuple[ast_nodes.StructFieldDecl | ast_nodes.AnonymousUnionDecl, ...],
  )
  assert_type(mjxmacro.MJMODEL_S, ast_nodes.StructDecl)
  assert_type(mjxmacro.MJMODEL, dict[str, tuple[str, str]])
  assert_type(mjxmacro.field, ast_nodes.StructFieldDecl)
  assert_type(mjxmacro.MJDATA['qpos'], tuple[str, str])
  parsed = type_parsing.parse_type('const mjModel*')
  assert_type(
    parsed, ast_nodes.ValueType | ast_nodes.PointerType | ast_nodes.ArrayType
  )
  type_parsing.parse_function_return_type('int (void)')
  assert_type(type_parsing.ARRAY_N_PATTERN, re.Pattern[str])


def codegen(node: Mapping[str, Any]) -> None:
  processor = ast_processor.process('ast.json', headers=['mujoco.h'])
  assert_type(processor, ast_processor.AstProcessor)
  assert_type(
    processor.exported_functions, Mapping[str, ast_nodes.FunctionDecl]
  )
  assert_type(processor.type_aliases, Mapping[str, str])
  processor.visit(node)
  processor.resolve()
  assert_type(formatter.format_as_python_code(processor.exported_enums), str)
  assert_type(formatter.MAX_LINE_WIDTH, int)
  assert_type(formatter.format_as_python_code({'a': [1, (2.0, None)]}), str)
  assert_type(formatter.SIMPLE_TYPES, frozenset[type[object]])
  generate.main(['generate'])


def contexts() -> None:
  ctx = glfw.GLContext(32, 32)
  ctx.make_current()
  assert_type(
    ctx.as_current(), contextlib._GeneratorContextManager[None, None, None]
  )
  with ctx.as_current():
    pass
  ctx.free()
  other = cgl.GLContext(32, 32)
  other.make_current()
  egl.GLContext(32, 32).free()
  osmesa.GLContext(32, 32).free()
  assert_type(osmesa.PYOPENGL_PLATFORM, str | None)
  assert_type(egl.EGL_ATTRIBUTES, tuple[int, ...])
  assert_type(egl_ext.EGL_CUDA_DEVICE_NV, int)
  assert_type(egl_ext.eglQueryDevicesEXT(), list[egl_ext._EGLDeviceEXT])
  assert_type(gl_context.GLContext(8, 8), gl_context.GLContext)
  assert renderer.Renderer is not None


def cgl_bindings_usage() -> None:
  assert_type(cgl_bindings.CGLContextObj(), ctypes.c_void_p)
  assert_type(
    cgl_bindings.CGLPixelFormatAttribute.CGLPFADepthSize.value, Literal[12]
  )
  assert_type(
    cgl_bindings.CGLOpenGLProfile(0x1000), cgl_bindings.CGLOpenGLProfile
  )
  pix = cgl_bindings.CGLPixelFormatObj()
  cgl_bindings.CGLSetCurrentContext(pix)
  cgl_bindings.CGLReleasePixelFormat(pix)
  assert_type(cgl_bindings.CGLError('x'), cgl_bindings.CGLError)


def egl_usage() -> None:
  assert_type(egl.EGL_DISPLAY, egl_ext.EGLDisplay | None)
  assert_type(egl.create_initialized_egl_device_display(), egl_ext.EGLDisplay)
  assert_type(egl_ext.EGL_NO_DISPLAY.address, int)
