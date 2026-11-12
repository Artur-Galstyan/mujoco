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

"""Static AST tracing to find MuJoCo Model and Data field usages."""

import dataclasses
from collections.abc import Callable, Sequence
from typing import Dict, Set, Tuple

@dataclasses.dataclass
class FieldInfo:
  param_source: str
  expected_type: str
  param_order: tuple[int, str]

@dataclasses.dataclass
class FieldUsage:
  model_fields: Sequence[str] = ...
  data_fields: Sequence[str] = ...
  data_out_fields: Sequence[str] = ...
  render_context_in_caller: bool = ...

def trace_function(
  fpath: str,
  fn: str,
  mjwarp_field_info: Dict[str, FieldInfo],
  visited_fns: Set[Tuple[str, str]] | None = ...,
) -> FieldUsage: ...
def get_mjwarp_field_info(
  src: str,
  get_cls_type_annotations: Callable[[str], Dict[str, Dict[str, str]]],
) -> Dict[str, FieldInfo]:
  """Return field info for mujoco_warp/_src/types.py."""

def get_mjx_warp_field_info(
  src: str,
  get_cls_type_annotations: Callable[[str], Dict[str, Dict[str, str]]],
) -> Dict[str, FieldInfo]:
  """Return field info for mjx/warp/types.py."""
