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

"""Static sysid consumer; ty checks this file but it must never run."""

import pathlib
from collections.abc import Sequence

import numpy as np
import scipy.optimize as scipy_optimize
from numpy import typing as npt
from typing_extensions import assert_type

import mujoco
from mujoco import sysid
from mujoco.sysid import model_modifier, parameter, signal_modifier
from mujoco.sysid._src import (
  io,
  optimize,
  plotting,
  signal_transform,
  timeseries,
  trajectory,
)

_F64 = npt.NDArray[np.float64]


def sysid_parameters(spec: mujoco.MjSpec, model: mujoco.MjModel) -> None:
  param = parameter.Parameter('mass', 1.0, 0.1, 10.0)
  assert_type(param.name, str)
  assert_type(param.value, _F64)
  assert_type(param.modifier, parameter.ModifierFn | None)
  assert_type(param.inertia_type, parameter.InertiaType | None)
  assert_type(param.size, int)
  assert_type(param.shape, tuple[int, ...])
  assert_type(param.as_vector(), _F64)
  assert_type(param.get_bounds(), tuple[_F64, _F64])
  assert_type(param.sample(np.random.default_rng(0)), _F64)
  param.update_from_vector(np.zeros(1))
  param.move_off_bound(fraction=0.1)

  pdict = sysid.ParameterDict({'mass': param})
  pdict.add(sysid.Parameter('k', [1.0, 2.0], [0.0, 0.0], [3.0, 3.0]))
  assert_type(pdict['mass'], sysid.Parameter)
  assert_type(pdict.keys(), list[str])
  assert_type(pdict.items(), list[tuple[str, sysid.Parameter]])
  assert_type(pdict.size, int)
  assert_type(pdict.copy(), sysid.ParameterDict)
  assert_type(pdict.move_off_bounds(), sysid.ParameterDict)
  assert_type(pdict.as_vector(include_frozen=True), _F64)
  assert_type(pdict.sample(), _F64)
  assert_type(pdict.get_bounds(), tuple[_F64, _F64])
  assert_type(pdict.get_non_frozen_parameter_names(), list[str])
  assert_type(
    pdict.compare_parameters(pdict.as_vector(), pdict.as_vector()), str
  )
  assert_type(
    sysid.ParameterDict.load_from_disk(pathlib.Path('p.yaml')),
    sysid.ParameterDict,
  )
  pdict.save_to_disk('p.yaml')
  pdict.update_from_vector(pdict.as_vector())

  inertia = sysid.body_inertia_param(
    spec, model, 'b', inertia_type=sysid.InertiaType.Pseudo
  )
  assert_type(inertia, sysid.Parameter)
  assert_type(sysid.InertiaType(0), sysid.InertiaType)


def sysid_model_modifier(spec: mujoco.MjSpec, model: mujoco.MjModel) -> None:
  pdict = sysid.ParameterDict()
  assert_type(sysid.remove_visuals(spec), mujoco.MjSpec)
  assert_type(sysid.apply_param_modifiers_spec(pdict, spec), mujoco.MjSpec)
  assert_type(sysid.apply_param_modifiers(pdict, spec), mujoco.MjModel)
  assert_type(model_modifier.apply_pgain(spec, 'a', 1.0), mujoco.MjSpec)
  assert_type(sysid.apply_dgain(spec, 'a', np.ones(1)), mujoco.MjSpec)
  assert_type(sysid.apply_pdgain(spec, 'a', np.ones(2)), mujoco.MjSpec)
  assert_type(
    model_modifier.get_actuator_pd_gains(model, 'a'), tuple[float, float]
  )
  assert_type(model_modifier.is_position_actuator(model.actuator('a')), bool)
  actuator = spec.actuator('a')
  assert actuator is not None
  assert_type(model_modifier.is_position_actuator(actuator), bool)
  assert_type(model_modifier.pi_from_body(spec, 'b'), _F64)
  assert_type(model_modifier.pi_from_theta(np.zeros(10)), _F64)
  assert_type(model_modifier.skew(np.zeros(3)), _F64)
  assert_type(
    model_modifier.apply_body_mass_ipos(spec, 'b', mass=1.0), mujoco.MjSpec
  )
  assert_type(
    sysid.apply_body_inertia(spec, 'b', parameter.Parameter('p', 1, 0, 2)), None
  )


def sysid_timeseries(model: mujoco.MjModel) -> None:
  times = np.arange(10, dtype=np.float64)
  data = np.zeros((10, 2))
  ts = sysid.TimeSeries.from_names(
    times, data, model, ['jp', ('jv', sysid.SignalType.MjSensor)]
  )
  assert_type(ts, sysid.TimeSeries)
  assert_type(ts.times, _F64)
  assert_type(ts.signal_mapping, timeseries.SignalMappingType | None)
  assert_type(
    ts.get_indices('jp'), tuple[sysid.SignalType, npt.NDArray[np.int_]]
  )
  assert_type(ts.resample(target_dt=0.5), sysid.TimeSeries)
  assert_type(ts.interpolate(0.5, method='zoh'), _F64)
  assert_type(ts.get(np.zeros(2)), tuple[_F64, _F64])
  assert_type(ts.dt_statistics(), dict[str, float])
  assert_type(ts.remove_from_beginning(1.0), sysid.TimeSeries)
  assert_type(len(ts), int)
  assert_type(sysid.TimeSeries.load_from_disk('a.npz'), sysid.TimeSeries)
  custom = sysid.TimeSeries.from_custom_map(
    times, data, ['a', ('b', 1, sysid.SignalType.CustomObs)]
  )
  assert_type(custom, sysid.TimeSeries)
  assert_type(sysid.TimeSeries.slice_by_name(ts, ['jp']), sysid.TimeSeries)
  assert_type(
    sysid.TimeSeries.compute_all_sensor_mapping(model),
    timeseries.SignalMappingType,
  )
  assert_type(timeseries.encode_signal_mapping(ts.signal_mapping or {}), str)
  assert_type(
    sysid.TimeSeries.compute_all_state_mappings(model),
    tuple[
      timeseries.SignalMappingType,
      timeseries.SignalMappingType,
      timeseries.SignalMappingType,
      timeseries.SignalMappingType,
    ],
  )
  ts.save_to_disk('a.npz')
  ts.save_to_csv(pathlib.Path('a.csv'))


def sysid_signals(model: mujoco.MjModel, ts: sysid.TimeSeries) -> None:
  pdict = sysid.ParameterDict()
  delay = pdict['delay']
  assert_type(sysid.get_sensor_indices(model, ['a', 'b'], sort=True), list[int])
  assert_type(signal_modifier.apply_bias(ts, 'jp', delay), sysid.TimeSeries)
  assert_type(sysid.apply_gain(ts, 'jp', delay), sysid.TimeSeries)
  assert_type(sysid.apply_delay(ts, 'jp', delay), sysid.TimeSeries)
  assert_type(sysid.apply_delayed_ts_window(ts, ts, 0.0, 0.1), sysid.TimeSeries)
  assert_type(
    sysid.apply_resample_and_delay(ts, ts.times, 0.0, {'jp': 0.1}),
    sysid.TimeSeries,
  )
  diff = sysid.weighted_diff(ts.data, ts.data, model, {'jp': 2.0})
  assert_type(diff, _F64)
  assert_type(sysid.normalize_residual(diff, ts.data), _F64)

  transform = sysid.SignalTransform(normalize=False)
  transform.delay('*_pos', delay)
  transform.gain('*_torque', delay, target='predicted')
  transform.bias('*', delay)
  transform.enable_sensors(['jp'])
  transform.set_sensor_weights({'jp': 1.0})
  assert_type(transform.normalize, bool)
  assert_type(
    transform.apply(pdict, ts, ts, model, False),
    tuple[_F64, sysid.TimeSeries, sysid.TimeSeries],
  )
  fn: sysid.ModifyResidualFn = transform.apply
  del fn


def sysid_trajectory(
  model: mujoco.MjModel, spec: mujoco.MjSpec, ts: sysid.TimeSeries
) -> None:
  state = sysid.create_initial_state(
    model, np.zeros(model.nq), qpos_names=['j']
  )
  assert_type(state, _F64)
  seqs = sysid.ModelSequences('m', spec, ['s'], [state], [ts], ts)
  assert_type(seqs.gt_model, mujoco.MjModel)
  assert_type(seqs.sequence_name, list[str])
  assert_type(seqs.initial_state, list[_F64])
  assert_type(seqs.control, list[sysid.TimeSeries])
  assert_type(seqs.measured_rollout, list[sysid.SystemTrajectory])
  assert_type(seqs[0], sysid.ModelSequences)
  assert_type(seqs[0:1], sysid.ModelSequences)

  data = mujoco.MjData(model)
  trajs = sysid.sysid_rollout([model], [data], [ts], [state])
  assert_type(trajs, Sequence[sysid.SystemTrajectory])
  traj = trajs[0]
  assert_type(traj.model, mujoco.MjModel)
  assert_type(traj.state, sysid.TimeSeries | None)
  assert_type(traj.initial_state, _F64)
  assert_type(traj.get_sensordata_slice('joint_vel'), _F64)
  assert_type(traj.sensordim, int)
  assert_type(traj.split(10), list[sysid.SystemTrajectory])
  assert_type(traj.render(fps=10), list[npt.NDArray[np.uint8]])
  assert_type(traj.replace(state=None), sysid.SystemTrajectory)
  assert_type(
    sysid.SystemTrajectory.load_from_disk(pathlib.Path('t.npz'), model),
    sysid.SystemTrajectory,
  )
  traj.check_compatible(allow_missing_sensors=True)
  traj.save_to_disk(pathlib.Path('t.npz'))
  assert_type(trajectory.timeseries2array(ts), tuple[_F64, _F64])
  assert_type(trajectory.sequence2array([state]), _F64)
  assert_type(
    plotting.render_rollout(
      model, data, np.zeros((1, 2, 3)), 30, light_pos=[0, 0, 1]
    ),
    list[npt.NDArray[np.uint8]],
  )


def sysid_residual_and_optimize(
  model: mujoco.MjModel, spec: mujoco.MjSpec, seqs: sysid.ModelSequences
) -> None:
  pdict = sysid.ParameterDict()
  x = pdict.as_vector()
  res, preds, records = sysid.residual(
    x, pdict, [seqs], sensor_weights={'jp': 1.0}
  )
  assert_type(res, list[_F64])
  assert_type(preds, list[list[sysid.TimeSeries]])
  assert_type(records, list[list[sysid.TimeSeries]])

  fn = sysid.build_residual_fn(models_sequences=[seqs], n_threads=1)
  fn_res, fn_preds, fn_records = fn(x, pdict, return_pred_all=True)
  assert_type(fn_res, list[_F64])
  assert_type(fn_preds, list[list[sysid.TimeSeries]])
  assert_type(fn_records, list[list[sysid.TimeSeries]])
  build: sysid.BuildModelFn = sysid.apply_param_modifiers
  del build
  single = sysid.model_residual(
    x,
    pdict,
    lambda p: sysid.apply_param_modifiers(p, spec),
    seqs.measured_rollout[0],
  )
  assert_type(
    single, tuple[_F64, list[sysid.TimeSeries], list[sysid.TimeSeries]]
  )
  ts = seqs.control[0]
  assert_type(
    sysid.construct_ts_from_defaults(ts, ts, ts),
    tuple[sysid.TimeSeries, sysid.TimeSeries],
  )

  opt_params, opt_result = sysid.optimize(
    pdict, fn, optimizer='scipy', max_iters=5
  )
  assert_type(opt_params, sysid.ParameterDict)
  assert_type(opt_result, scipy_optimize.OptimizeResult)
  assert_type(
    optimize.optimize(pdict, fn, check_conditioning=True)[0],
    sysid.ParameterDict,
  )
  cov, intervals = sysid.calculate_intervals(res, np.zeros((4, 2)))
  assert_type(cov, _F64)
  assert_type(intervals, _F64)
  sysid.save_results('out', [seqs], pdict, opt_params, opt_result, fn)
  io.save_results(
    pathlib.Path('out'), [seqs], pdict, opt_params, opt_result, fn
  )
  _ = signal_transform.SignalTransform
