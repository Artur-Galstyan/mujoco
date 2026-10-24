"""Type-check the official 3.17 batch modules; do not run this consumer."""

from typing import TypeAlias

import numpy as np
from numpy import typing as npt
from typing_extensions import assert_type

import mujoco
from mujoco import _batch, batch

_NumArray: TypeAlias = npt.NDArray[np.float32] | npt.NDArray[np.float64]
_FieldArray: TypeAlias = (
  _NumArray
  | npt.NDArray[np.int32]
  | npt.NDArray[np.uint8]
  | npt.NDArray[np.bool_]
  | npt.NDArray[np.int8]
  | npt.NDArray[np.uint64]
)


def check_batch_types(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  sims = batch.Batch(model, nsim=2, nthread=1, persistent=True)
  assert_type(sims.model, mujoco.MjModel)
  assert_type(sims.nsim, int)
  assert_type(sims.nthread, int)
  assert_type(sims.nstate, int)
  assert_type(sims.persistent, bool)
  assert_type(sims.state, _NumArray)
  assert_type(sims.warning, npt.NDArray[np.int32])
  assert_type(sims.status, npt.NDArray[np.int32])
  assert_type(sims.error(0), str)
  assert_type(batch.DATA_FIELDS, frozenset[str])
  assert_type(batch.MODEL_FIELDS, frozenset[str])

  assert_type(sims.bind('qpos'), _NumArray)
  assert_type(sims.bind('ncon'), npt.NDArray[np.int32])
  assert_type(sims.bind('eq_active'), _NumArray)
  assert_type(sims.bind('xpos'), _NumArray)
  assert_type(sims.bind('body_awake'), npt.NDArray[np.int32])
  assert_type(sims.bind('bvh_active'), npt.NDArray[np.bool_])
  assert_type(sims.bvh_active, npt.NDArray[np.bool_])
  assert_type(sims.bind('flex_hessian_valid'), npt.NDArray[np.bool_])
  assert_type(sims.bind('plugin_data'), npt.NDArray[np.uint64])
  assert_type(sims.plugin_data, npt.NDArray[np.uint64])
  assert_type(sims.qpos, _NumArray)
  sims.ctrl = np.zeros((2, model.nu))
  assert_type(sims.expand('body_mass'), _NumArray)
  assert_type(sims.expand('body_parentid'), npt.NDArray[np.int32])
  assert_type(sims.expand('body_simple'), npt.NDArray[np.uint8])
  assert_type(sims.expand('geom_rgba'), npt.NDArray[np.float32])
  assert_type(sims.expand('plugin_attr'), npt.NDArray[np.int8])
  assert_type(sims.expand('actuator_actearly'), npt.NDArray[np.bool_])
  assert_type(sims.joint('j').id, int)
  assert_type(sims.joint('j').name, str)
  assert_type(sims.joint('j').qpos, _NumArray)
  assert_type(sims.body('b').mass, _NumArray)
  assert_type(sims.body('b').simple, npt.NDArray[np.uint8])
  assert_type(sims.geom('g').rgba, npt.NDArray[np.float32])
  assert_type(sims.actuator(0).ctrl, _NumArray)
  assert_type(sims.actuator(0).force, _NumArray)
  assert_type(sims.body('b').xpos, _NumArray)
  assert_type(sims.camera(0).xmat, _NumArray)
  assert_type(sims.equality(0).data, _NumArray)
  assert_type(sims.geom('g').xpos, _NumArray)
  assert_type(sims.joint('j').range, _NumArray)
  assert_type(sims.keyframe(0).time, _NumArray)
  assert_type(sims.light(0).ambient, npt.NDArray[np.float32])
  assert_type(sims.material(0).texid, npt.NDArray[np.int32])
  assert_type(sims.numeric(0).data, _NumArray)
  assert_type(sims.pair(0).signature, npt.NDArray[np.int32])
  assert_type(sims.sensor(0).data, _NumArray)
  assert_type(sims.site(0).sameframe, npt.NDArray[np.uint8])
  assert_type(sims.tendon(0).J_rowadr, npt.NDArray[np.int32])
  assert_type(sims.tendon(0)._damping, _NumArray)
  sims.joint('j').qpos[:, 0] = [0, 1]
  sims.body('b').mass = [1, 2]
  for view in (
    sims.actuator(0),
    sims.camera(0),
    sims.equality(0),
    sims.keyframe(0),
    sims.light(0),
    sims.material(0),
    sims.numeric(0),
    sims.pair(0),
    sims.sensor(0),
    sims.site(0),
    sims.tendon(0),
  ):
    assert_type(view.id, int)

  ids = np.array([0], dtype=np.int32)
  mask = np.array([True, False])
  assert_type(sims.step(ids, nstep=2, stage=mujoco.mjtStage.mjSTAGE_VEL), None)
  assert_type(sims.forward(mask, stage=mujoco.mjtStage.mjSTAGE_ACC), None)
  assert_type(sims.reset(keyframe=0, stage=mujoco.mjtStage.mjSTAGE_NONE), None)
  assert_type(sims.set_const(ids), None)
  assert_type(sims.get_model(0), mujoco.MjModel)
  assert_type(sims.get_model(0, model), mujoco.MjModel)
  assert_type(sims.get_data(0), mujoco.MjData)
  assert_type(sims.get_data(0, data), mujoco.MjData)
  assert_type(sims.set_models([model, model]), None)
  sims.set_models((model, model))
  sims.set_models(model)
  sims.set_models(None)
  record: list[batch.RecordItem] = ['qpos', mujoco.mjtState.mjSTATE_QVEL]
  assert_type(
    sims.rollout(
      2,
      control_spec=mujoco.mjtState.mjSTATE_CTRL,
      record=record,
      out={'qpos': np.zeros((2, 2, model.nq))},
    ),
    dict[batch.RecordItem, _FieldArray],
  )
  assert_type(
    sims.rollout(1, record=('plugin_data',))['plugin_data'],
    npt.NDArray[np.uint64],
  )
  assert_type(sims.jac(1, np.zeros(3), ids), tuple[_NumArray, _NumArray])
  assert_type(
    sims.ray([0, 0, 1], [0, 0, -1]), tuple[_NumArray, npt.NDArray[np.int32]]
  )
  assert_type(
    sims.ray([0, 0, 1], [0, 0, -1], normal=True),
    tuple[_NumArray, npt.NDArray[np.int32], _NumArray],
  )
  try:
    sims.step()
  except batch.SimulationError as error:
    assert_type(error.ids, npt.NDArray[np.int32] | npt.NDArray[np.int64])


def check_native_types(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  raw = _batch.Batch(model, nsim=2, nthread=1)
  ids = np.array([0, 1], dtype=np.int32)
  num = np.zeros((2, 1, 3), dtype=np.float64)
  bodies = np.zeros(2, dtype=np.int32)
  assert_type(raw.nsim, int)
  assert_type(raw.nthread, int)
  assert_type(raw.nstate, int)
  assert_type(raw.persistent, bool)
  assert_type(raw.state(), _NumArray)
  assert_type(
    raw.state_component(mujoco.mjtState.mjSTATE_QPOS), tuple[int, int]
  )
  assert_type(raw.warning(), npt.NDArray[np.int32])
  assert_type(raw.status(), npt.NDArray[np.int32])
  assert_type(raw.error(0), str)
  assert_type(raw.output('xpos'), npt.NDArray[np.uint8] | None)
  assert_type(raw.expand('body_mass'), npt.NDArray[np.uint8] | None)
  assert_type(raw.expanded('body_mass'), npt.NDArray[np.uint8] | None)
  assert_type(raw.step(ids, stage=mujoco.mjtStage.mjSTAGE_NONE), int)
  assert_type(raw.forward(ids, stage=mujoco.mjtStage.mjSTAGE_ACC), int)
  assert_type(raw.reset(ids, bodies, stage=mujoco.mjtStage.mjSTAGE_ACC), int)
  assert_type(raw.set_const(ids), int)
  assert_type(raw.get_model(0, model), None)
  assert_type(raw.get_data(0, data), None)
  assert_type(raw.set_models((model, model)), None)
  assert_type(raw.set_models([model, model]), None)
  assert_type(
    raw.rollout(ids, 1, None, mujoco.mjtState.mjSTATE_CTRL, [('qpos', 0, num)]),
    int,
  )
  assert_type(
    raw.rollout(
      ids,
      1,
      None,
      mujoco.mjtState.mjSTATE_CTRL,
      [('plugin_data', 0, np.zeros((2, 1, 0), dtype=np.uint64))],
    ),
    int,
  )
  assert_type(raw.jac(ids, bodies, num[:, 0, :], num, None), int)
  assert_type(
    raw.ray(
      ids,
      num,
      num,
      None,
      True,
      bodies[:, None],
      num[:, :, 0],
      bodies[:, None],
      None,
    ),
    int,
  )
