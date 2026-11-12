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

"""Automated insights report section."""

from typing import Any

from mujoco.sysid._src import parameter
from mujoco.sysid.report.sections.base import ReportSection

class AutomatedInsights(ReportSection):
  """Analyzes identification results and generates automated insights."""

  def __init__(
    self,
    title: str,
    parameter_dict: parameter.ParameterDict,
    anchor: str = 'insights',
    collapsible: bool = True,
  ) -> None: ...
  @property
  def template_filename(self) -> str: ...
  @property
  def title(self) -> str: ...
  @property
  def anchor(self) -> str: ...
  def get_context(self) -> dict[str, Any]: ...
