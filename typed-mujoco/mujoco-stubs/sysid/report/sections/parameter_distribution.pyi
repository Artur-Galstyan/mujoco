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

"""Parameter distribution report section."""

from typing import Any

import numpy as np
from numpy import typing as npt

from mujoco.sysid._src import parameter
from mujoco.sysid.report.sections.base import ReportSection

class ParameterDistribution(ReportSection):
  """Displays the identified parameters relative to their bounds."""

  def __init__(
    self,
    title: str,
    opt_params: parameter.ParameterDict,
    initial_params: parameter.ParameterDict,
    confidence_intervals: npt.NDArray[np.number[Any]] | None = None,
    height_per_param: int = 60,
    anchor: str = '',
    collapsible: bool = True,
  ) -> None: ...
  @property
  def template_filename(self) -> str: ...
  @property
  def title(self) -> str: ...
  @property
  def anchor(self) -> str: ...
  def header_includes(self) -> set[str]: ...
  def get_context(self) -> dict[str, Any]: ...
