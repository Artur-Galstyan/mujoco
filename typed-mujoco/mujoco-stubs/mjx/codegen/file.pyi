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

"""File tools."""

from typing import Dict

from etils import epath

LICENSE_TEXT: str

def get_base_path() -> epath.Path:
  """Resolves the base workspace path."""

def format_file(target_fpath: epath.Path) -> None:
  """Formats a Python file."""

def write_license(target_fpath: epath.Path) -> None:
  """Writes license to the target file."""

def get_cls_type_annotations(src: str) -> Dict[str, Dict[str, str]]:
  """Return classes with their field annotation strings from source code."""
