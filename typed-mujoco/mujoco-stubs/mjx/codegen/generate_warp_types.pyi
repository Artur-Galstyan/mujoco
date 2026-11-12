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

"""Generate types for MJX warp integration."""

from collections.abc import Sequence
from typing import Any, Dict, Optional

from etils import epath

def write_header(target_fpath: epath.Path) -> None:
  """Writes imports and pre-defined class definitions to types.py."""

def write_nested_dataclass(
  target_fpath: epath.Path, cls: type[object]
) -> None: ...
def write_core_cls(
  cls_name: str,
  target_fpath: epath.Path,
  mjx_types_fpath: epath.Path,
  flatten_fields: bool = ...,
  set_diff: bool = ...,
  extra_annotations: dict[str, type] | None = ...,
  defaults: Optional[Dict[str, Any]] = ...,
) -> None:
  """Writes a core API class (e.g. Model/Data/Option)."""

def write_register_vmappable(target_fpath: epath.Path) -> None:
  """Writes register_vmappable to types.py."""

def write_ndim_annotations(target_fpath: epath.Path) -> None:
  """Writes a dictionary of ndim for every warp field."""

def write_nworld_leading_dim(target_fpath: epath.Path) -> None:
  """Writes a dictionary of which MJW fields are batched."""

def main(argv: Sequence[str]) -> None: ...
