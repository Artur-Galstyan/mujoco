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

"""Static consumer of the installed typed-mujoco stubs and mujoco runtime."""

from typing import SupportsInt, TypeAlias

import numpy as np
from numpy import typing as npt
from typing_extensions import assert_type

import mujoco
from mujoco import _simulate, gl_context, renderer, rollout, viewer
from mujoco.rendering.classic import gl_context as classic_gl_context
from mujoco.rendering.classic import renderer as classic_renderer

_NumArray: TypeAlias = npt.NDArray[np.float32] | npt.NDArray[np.float64]


def accepts_enum(value: SupportsInt) -> None:
  pass


def check_types() -> None:
  model = mujoco.MjModel.from_xml_string('<mujoco/>')
  assert_type(model, mujoco.MjModel)
  data = mujoco.MjData(model)
  assert_type(data, mujoco.MjData)
  assert_type(model.nbody, int)
  assert_type(data.time, float)
  assert_type(data.qpos, _NumArray)
  assert_type(model.geom('geom').size, _NumArray)
  assert_type(model.body('body').pos, _NumArray)
  assert_type(data.joint('joint').qpos, _NumArray)
  assert_type(model.jnt('joint').range, _NumArray)
  assert_type(model.cam('camera').pos, _NumArray)
  assert_type(data.jnt('joint').qpos, _NumArray)
  assert_type(data.cam('camera').xpos, _NumArray)
  assert_type(model.opt.gravity, _NumArray)
  assert_type(model.efmC_perm, npt.NDArray[np.int32])
  assert_type(model.nefmCvert, int)
  assert_type(model.tex_contenttypeadr, npt.NDArray[np.int32])
  model.efmC_perm = model.efmC_perm
  model.opt = model.opt
  model.vis = model.vis
  model.stat = model.stat
  data.qpos = [0]
  model.opt.gravity = [0, 0, -9.81]

  mujoco.mj_step(model, data, nstep=2)
  mujoco.mj_forward(model, data)
  flags = mujoco.mjtState.mjSTATE_QPOS | mujoco.mjtState.mjSTATE_QVEL
  assert_type(flags, int)
  assert_type(mujoco.mj_stateSize(model, flags), int)
  assert_type(mujoco.mjtState.mjSTATE_QPOS.value, int)
  accepts_enum(mujoco.mjtJoint.mjJNT_HINGE)
  accepts_enum(mujoco.mjtActuator.mjACTUATOR_POSITION)
  accepts_enum(mujoco.mjtEmbed.mjEMBED_AUTO)
  accepts_enum(mujoco.mjtFlexSection.mjFLEXSECTION_BOX)
  accepts_enum(mujoco.mjtPassthrough.mjPASSTHROUGH_TEXTURES)
  accepts_enum(mujoco.mjtScaling.mjSCALING_GEOMETRY)
  accepts_enum(mujoco.mjtCtrlInput.mjINPUT_PRESSURE)
  accepts_enum(mujoco.mjtConDataField.mjCONDATA_LINVEL)
  accepts_enum(mujoco.mjtMeshBuiltin.mjMESH_BUILTIN_GRID)
  assert_type(mujoco.mjSC_IN | mujoco.mjSC_OUT | mujoco.mjSC_INOUT, int)
  assert_type([0, 1, 2, 3][mujoco.mjtJoint.mjJNT_HINGE], int)
  assert_type(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'body'), int)
  jacp = np.zeros((3, model.nv), dtype=np.float64)
  jacr = np.zeros((3, model.nv), dtype=np.float64)
  mujoco.mj_jac(model, data, jacp, jacr, np.zeros(3), 0)

  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name='body', pos=[0.0, 0.0, 0.0])
  assert_type(body, mujoco.MjsBody)
  assert_type(body.fuse, int)
  assert_type(body.add_geom(name='geom', size=[0.1, 0.0, 0.0]), mujoco.MjsGeom)
  joint = body.add_freejoint(name='joint')
  assert_type(joint, mujoco.MjsJoint)
  assert_type(spec.body('body'), mujoco.MjsBody | None)
  assert_type(body.add_frame(scale=[1, 1, 1]).scale, npt.NDArray[np.float64])
  assert_type(spec.compiler.passthrough, mujoco.mjtPassthrough)
  assert_type(spec.compile(), mujoco.MjModel)
  assert_type(
    spec.add_actuator(type=mujoco.mjtActuator.mjACTUATOR_POSITION),
    mujoco.MjsActuator,
  )
  roundtrip_spec = mujoco.MjSpec()
  roundtrip_spec.copy_back(roundtrip_spec.compile())
  assert_type(mujoco.mj_embedAssets(model), mujoco.MjModel)
  texture_spec = mujoco.MjSpec()
  texture = texture_spec.add_texture(
    name='checker',
    type=mujoco.mjtTexture.mjTEXTURE_2D,
    width=1,
    height=1,
    nchannel=3,
    data=[255, 0, 0],
    embed=mujoco.mjtEmbed.mjEMBED_AUTO,
  )
  texture_model = texture_spec.compile()
  assert_type(texture_model.bind(texture).colorspace, npt.NDArray[np.int32])
  assert_type(texture_model.bind(texture).pathadr, npt.NDArray[np.int32])
  assert_type(texture.embed, mujoco.mjtEmbed)
  assert_type(texture_model.bind([texture]).colorspace, list[np.int32])
  assert_type(texture_model.bind([texture]).contenttypeadr, list[np.int32])
  assert_type(texture_model.bind([texture]).pathadr, list[np.int32])
  assert_type(
    mujoco.mj_loadTexture(texture_model, 0), tuple[npt.NDArray[np.uint8], int]
  )
  assert_type(spec.attach(mujoco.MjSpec(), site='anchor'), mujoco.MjsFrame)
  assert_type(
    spec.attach(mujoco.MjSpec(), frame=body.add_frame(name='mount')),
    mujoco.MjsFrame,
  )
  assert_type(spec.attach(mujoco.MjSpec(), frame='mount'), mujoco.MjsFrame)

  assert_type(spec.encode('model.mjb', model=model), int)
  assert_type(
    spec.add_tendon().wrap_site('anchor').target,
    mujoco.MjsSite | mujoco.MjsGeom | mujoco.MjsJoint | None,
  )
  assert_type(spec.add_tendon().wrap_site('anchor').coef, float | None)
  element: mujoco.MjStruct = body
  assert_type(model.bind(element).pos, _NumArray)
  assert_type(data.bind(body).xpos, _NumArray)
  assert_type(model.bind([body]).pos, list[np.float32 | np.float64])
  assert_type(data.bind([body]).xpos, list[np.float32 | np.float64])
  assert_type(data.bind([joint]).cdof, list[_NumArray])
  assert_type(mujoco.MjLogMessage().body, str | None)
  assert_type(mujoco.MjLogMessage().file, str | None)
  mujoco.to_zip(spec, 'model.zip')
  spec.to_zip('model.zip')
  assert_type(mujoco.from_zip('model.zip'), mujoco.MjSpec)
  assert_type(mujoco.MjSpec.from_zip('model.zip'), mujoco.MjSpec)

  mujoco.set_mjcb_time(lambda: 0.0)
  mujoco.set_mjcb_time(None)
  mujoco.set_mjcb_control(lambda m, d: None)
  mujoco.set_mjcb_control(None)

  assert_type(renderer.Renderer(model), classic_renderer.Renderer)
  assert_type(mujoco.Renderer(model), classic_renderer.Renderer)
  context_type: type[classic_gl_context.GLContext] = gl_context.GLContext
  context_type(64, 64)
  assert_type(
    rollout.rollout(model, data, np.zeros((1, 1)), nstep=1),
    tuple[_NumArray, _NumArray],
  )
  assert_type(viewer.launch_passive(model, data), viewer.Handle)
  handle: viewer.Handle = viewer.launch_passive(model, data)
  with handle.lock():
    handle.sync()


def check_simulate_types(simulator: _simulate.Simulate) -> None:
  assert_type(simulator.m, mujoco.MjModel | None)
  assert_type(simulator.d, mujoco.MjData | None)
  simulator.set_texts([(0, 0, 'title', 'text')])
  simulator.set_images(
    [(mujoco.MjrRect(0, 0, 32, 32), np.zeros((32, 32, 3), dtype=np.uint8))]
  )
