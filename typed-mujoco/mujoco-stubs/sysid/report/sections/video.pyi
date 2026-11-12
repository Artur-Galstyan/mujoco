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

"""Video report section for system identification results."""

import os
import pathlib
from collections.abc import Callable, Sequence
from typing import Any

import mujoco
from mujoco.sysid._src import parameter
from mujoco.sysid._src.trajectory import SystemTrajectory
from mujoco.sysid.report.sections.base import ReportSection

def spec_apply(
  spec: mujoco.MjSpec,
  attrs: Sequence[str],
  values: Sequence[Any],
) -> None: ...
def generate_video_from_trajectories(
  initial_params: parameter.ParameterDict,
  opt_params: parameter.ParameterDict,
  _build_model: Callable[
    [parameter.ParameterDict, mujoco.MjSpec], mujoco.MjModel
  ],
  trajectories: list[SystemTrajectory],
  model_spec: mujoco.MjSpec,
  output_filepath: os.PathLike[str],
  render_initial: bool = True,
  render_nominal: bool = True,
  render_opt: bool = True,
  height: int = 480,
  width: int = 640,
  fovy: float = 60,
  camera: str | int = -1,
  fps: int = 60,
) -> pathlib.Path: ...

class VideoPlayer(ReportSection):
  """A report section to embed and display a video file."""

  def __init__(
    self,
    title: str,
    video_filepath: pathlib.Path,
    anchor: str = '',
    width: int | str = 800,
    height: int | None = 450,
    autoplay: bool = False,
    controls: bool = True,
    muted: bool = False,
    loop: bool = True,
    caption: str = ...,
    collapsible: bool = True,
  ) -> None: ...
  @property
  def title(self) -> str: ...
  @property
  def anchor(self) -> str: ...
  @property
  def template_filename(self) -> str: ...
  def header_includes(self) -> set[str]: ...
  def get_context(self) -> dict[str, Any]: ...
