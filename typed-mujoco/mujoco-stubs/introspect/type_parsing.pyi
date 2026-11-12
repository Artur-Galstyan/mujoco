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

"""Functions for parsing C type declarations."""

import re
from typing import TypeAlias

from mujoco.introspect import ast_nodes

_ParsedType: TypeAlias = (
  ast_nodes.ValueType | ast_nodes.PointerType | ast_nodes.ArrayType
)

ARRAY_EXTENTS_PATTERN: re.Pattern[str]
ARRAY_N_PATTERN: re.Pattern[str]
STARTS_WITH_CONST_PATTERN: re.Pattern[str]
ENDS_WITH_CONST_PATTERN: re.Pattern[str]

def parse_type(type_name: str) -> _ParsedType:
  """Parses a string that represents a C type into an AST node."""

def parse_function_return_type(type_name: str) -> _ParsedType: ...
