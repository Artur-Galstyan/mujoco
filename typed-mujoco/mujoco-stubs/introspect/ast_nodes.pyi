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

"""Classes that roughly correspond to Clang AST node types."""

import dataclasses
import re
from collections.abc import Sequence

VALID_TYPE_NAME_PATTERN: re.Pattern[str]
C_INVALID_TYPE_NAMES: frozenset[str]

@dataclasses.dataclass
class ValueType:
  """Represents a C type that is neither a pointer type nor an array type."""

  name: str
  is_const: bool = False
  is_volatile: bool = False
  nullable: bool = False

  def __init__(
    self,
    name: str,
    is_const: bool = False,
    is_volatile: bool = False,
    nullable: bool = False,
  ) -> None: ...
  def decl(self, name_or_decl: str | None = None) -> str: ...

@dataclasses.dataclass
class ArrayType:
  """Represents a C array type."""

  inner_type: ValueType | PointerType
  extents: tuple[int, ...]
  nullable: bool = False

  def __init__(
    self, inner_type: ValueType | PointerType, extents: Sequence[int]
  ) -> None: ...
  def decl(self, name_or_decl: str | None = None) -> str: ...

@dataclasses.dataclass
class PointerType:
  """Represents a C pointer type."""

  inner_type: ValueType | ArrayType | PointerType
  nullable: bool = False
  is_const: bool = False
  is_volatile: bool = False
  is_restrict: bool = False

  def decl(self, name_or_decl: str | None = None) -> str:
    """Creates a string that declares an object of this type."""

@dataclasses.dataclass
class FunctionParameterDecl:
  """Represents a parameter in a function declaration."""

  name: str
  type: ValueType | ArrayType | PointerType
  nullable: bool = False

  @property
  def decltype(self) -> str: ...

@dataclasses.dataclass
class FunctionDecl:
  """Represents a function declaration."""

  name: str
  return_type: ValueType | ArrayType | PointerType
  parameters: tuple[FunctionParameterDecl, ...]
  doc: str

  def __init__(
    self,
    name: str,
    return_type: ValueType | ArrayType | PointerType,
    parameters: Sequence[FunctionParameterDecl],
    doc: str,
  ) -> None: ...
  @property
  def decltype(self) -> str: ...

@dataclasses.dataclass
class EnumDecl:
  """Represents an enum declaration."""

  name: str
  declname: str
  values: dict[str, int]

  def __init__(
    self, name: str, declname: str, values: dict[str, int]
  ) -> None: ...

@dataclasses.dataclass
class StructFieldDecl:
  """Represents a field in a struct or union declaration."""

  name: str
  type: (
    ValueType
    | ArrayType
    | PointerType
    | AnonymousStructDecl
    | AnonymousUnionDecl
  )
  doc: str
  array_extent: tuple[str | int, ...] | None = None

  @property
  def decltype(self) -> str: ...

@dataclasses.dataclass
class AnonymousStructDecl:
  """Represents an anonymous struct declaration."""

  fields: tuple[StructFieldDecl | AnonymousUnionDecl, ...]

  def __init__(self, fields: Sequence[StructFieldDecl]) -> None: ...
  def decl(self, name_or_decl: str | None = None) -> str: ...

class AnonymousUnionDecl(AnonymousStructDecl):
  """Represents an anonymous union declaration."""

  def decl(self, name_or_decl: str | None = None) -> str: ...

@dataclasses.dataclass
class StructDecl:
  """Represents a struct declaration."""

  name: str
  declname: str
  fields: tuple[StructFieldDecl | AnonymousUnionDecl, ...]

  def __init__(
    self,
    name: str,
    declname: str,
    fields: Sequence[StructFieldDecl | AnonymousUnionDecl],
  ) -> None: ...
  def decl(self, name_or_decl: str | None = None) -> str: ...
