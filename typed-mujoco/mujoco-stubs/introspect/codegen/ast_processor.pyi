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

"""Parses MuJoCo enums, structs, and functions from a Clang AST JSON dump."""

from collections.abc import Mapping
from typing import Any, TypeAlias

from mujoco.introspect import ast_nodes

ClangJsonNode: TypeAlias = Mapping[str, Any]

class AstProcessor:
  """A Clang AST JSON node visitor for."""

  def __init__(
    self, raw_headers: Mapping[str, str], excluded: list[str]
  ) -> None: ...
  @property
  def exported_enums(self) -> Mapping[str, ast_nodes.EnumDecl]: ...
  @property
  def exported_functions(self) -> Mapping[str, ast_nodes.FunctionDecl]: ...
  @property
  def exported_structs(self) -> Mapping[str, ast_nodes.StructDecl]: ...
  @property
  def type_aliases(self) -> Mapping[str, str]: ...
  def visit(self, node: ClangJsonNode) -> None:
    """Visits a JSON node and stores information about API elements."""
  def resolve(self) -> None:
    """Replaces anonymous struct placeholders with corresponding decl."""

def process(
  input_json: str,
  headers: list[str] | None = None,
  excluded: list[str] | None = None,
) -> AstProcessor:
  """Parses a Clang AST JSON file into individual AST types."""
