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

"""Utility for formatting AST node as Python code."""

from collections.abc import Mapping, Sequence
from typing import TypeAlias

from mujoco.introspect import ast_nodes

_Formattable: TypeAlias = (
  int
  | float
  | str
  | bytes
  | None
  | ast_nodes.ValueType
  | ast_nodes.ArrayType
  | ast_nodes.PointerType
  | ast_nodes.FunctionParameterDecl
  | ast_nodes.FunctionDecl
  | ast_nodes.EnumDecl
  | ast_nodes.StructFieldDecl
  | ast_nodes.AnonymousStructDecl
  | ast_nodes.StructDecl
  | Mapping[str, _Formattable]
  | Sequence[_Formattable]
)

INDENT_WIDTH: int
MAX_LINE_WIDTH: int
SIMPLE_TYPES: frozenset[type[object]]

def format_as_python_code(obj: _Formattable) -> str:
  """Formats an AST node object as well-indented Python code."""
