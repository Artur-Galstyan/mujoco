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

"""HTML report builder using Jinja2 templates."""

import os
from typing import Any

from mujoco.sysid.report.sections.base import ReportSection

TEMPLATE_DIR: str

class ReportBuilder:
  """Assembles report sections into a complete HTML report."""

  def __init__(
    self, title: str, global_context: dict[str, Any] | None = None
  ) -> None: ...
  def add_section(self, section: ReportSection) -> None: ...
  def build(self) -> str: ...
  def save(self, path: str | os.PathLike[str]) -> None: ...
