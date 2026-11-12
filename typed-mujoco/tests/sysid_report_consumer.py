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

"""Static sysid.report consumer; ty checks this file but it must never run."""

import os
import pathlib
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import scipy.optimize as scipy_optimize
from numpy import typing as npt
from typing_extensions import assert_type

import mujoco
from mujoco.sysid._src import parameter
from mujoco.sysid._src.residual import BuildModelFn
from mujoco.sysid._src.timeseries import TimeSeries
from mujoco.sysid._src.trajectory import ModelSequences, SystemTrajectory
from mujoco.sysid.report import builder, defaults, utils
from mujoco.sysid.report.sections import (
  base,
  covariance,
  group,
  insights,
  optimization_trace,
  parameter_distribution,
  parameters,
  row,
  signals,
  video,
)


def utils_and_builder() -> None:
  assert_type(utils.plotly_script_tag(), str)
  assert_type(utils.get_text_color('#ffffff'), str)
  assert_type(builder.TEMPLATE_DIR, str)
  report = builder.ReportBuilder('title', {'key': 1})
  assert_type(report.build(), str)
  assert_type(report.add_section(insights_section()), None)
  report.save(pathlib.Path('report.html'))
  report.save('report.html')


def insights_section() -> insights.AutomatedInsights:
  params = parameter.ParameterDict()
  section = insights.AutomatedInsights('Log', params)
  assert_type(section.title, str)
  assert_type(section.anchor, str)
  assert_type(section.collapsible, bool)
  assert_type(section.is_open, bool)
  assert_type(section.template_filename, str)
  assert_type(section.get_context(), dict[str, Any])
  assert_type(section.header_includes(), Iterable[str])
  return section


def sections(
  model: mujoco.MjModel,
  params: parameter.ParameterDict,
  initial: parameter.ParameterDict,
  ts: TimeSeries,
) -> None:
  cov = covariance.Covariance('Cov', np.eye(2), params, anchor='cov')
  assert_type(cov.header_includes(), set[str])
  assert_type(cov.get_context(), dict[str, Any])

  table = parameters.ParametersTable('Params', params, initial)
  assert_type(table.template_filename, str)
  assert_type(table.header_includes(), set[str])

  dist = parameter_distribution.ParameterDistribution(
    'Dist', params, initial, confidence_intervals=np.zeros(2)
  )
  assert_type(dist.title, str)

  trace = optimization_trace.OptimizationTrace(
    'Trace',
    [1.0, 0.5],
    [np.zeros(2), [0.0, 1.0]],
    bounds=([0.0, 0.0], np.ones(2)),
    param_names=['a', 'b'],
  )
  assert_type(trace.header_sections(), set[str])

  sig = signals.SignalReport('Sig', model, ts_dict={'control': ts})
  assert_type(sig.header_includes(), set[str])

  player = video.VideoPlayer('Video', pathlib.Path('v.mp4'), height=None)
  assert_type(player.get_context(), dict[str, Any])

  grp = group.GroupSection('Group', [cov, table, sig])
  assert_type(grp.sections, list[base.ReportSection])
  assert_type(grp.header_includes(), set[str])

  rw = row.RowSection('Row', [player], description='d')
  assert_type(rw.sections, list[base.ReportSection])
  assert_type(rw.template_filename, str)


def video_functions(
  initial: parameter.ParameterDict,
  opt: parameter.ParameterDict,
  build_model: Callable[
    [parameter.ParameterDict, mujoco.MjSpec], mujoco.MjModel
  ],
  trajectories: list[SystemTrajectory],
  spec: mujoco.MjSpec,
  path: os.PathLike[str],
) -> None:
  out = video.generate_video_from_trajectories(
    initial, opt, build_model, trajectories, spec, path, fps=30
  )
  assert_type(out, pathlib.Path)
  assert_type(video.spec_apply(spec, ['rgba'], [[1, 0, 0, 1]]), None)


def default(
  seqs: list[ModelSequences],
  initial: parameter.ParameterDict,
  opt: parameter.ParameterDict,
  result: scipy_optimize.OptimizeResult,
  build_model: BuildModelFn,
  residual_fn: Callable[
    ...,
    tuple[
      list[npt.NDArray[np.float64]],
      list[list[TimeSeries]],
      list[list[TimeSeries]],
    ],
  ],
) -> None:
  rb = defaults.default_report(
    seqs,
    initial,
    opt,
    residual_fn,
    result,
    save_path=pathlib.Path('out'),
    build_model=build_model,
  )
  assert_type(rb, builder.ReportBuilder)


def numpy_dtypes(params: parameter.ParameterDict) -> None:
  dist = parameter_distribution.ParameterDistribution(
    'Dist',
    params,
    params,
    confidence_intervals=np.zeros(2, dtype=np.float32),
  )
  assert_type(dist.get_context(), dict[str, Any])
  trace = optimization_trace.OptimizationTrace(
    'Trace',
    [1.0],
    [np.zeros(2)],
    bounds=(np.zeros(2, dtype=np.int64), np.ones(2, dtype=np.float32)),
  )
  assert_type(trace.get_context(), dict[str, Any])
