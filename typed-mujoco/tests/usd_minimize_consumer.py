# Copyright 2026 DeepMind Technologies Limited
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS.
"""Static minimize, msh2obj and USD consumer; ty checks it, it never runs."""

import argparse
import pathlib
from typing import Any

import numpy as np
from numpy import typing as npt
from typing_extensions import assert_type

import mujoco
import mujoco._usd
import mujoco.usd.exporter
from mujoco import minimize, msh2obj
from mujoco.usd import (
  MjcPhysics,
  camera,
  demo,
  exporter,
  layer_sink,
  lights,
  objects,
  shapes,
  utils,
  writer,
)


class Squared(minimize.Norm):
  def value(self, r: npt.NDArray[np.float64]) -> float:
    return float(np.sum(r**2))

  def grad_hess(
    self, r: npt.NDArray[np.float64], proj: npt.NDArray[np.float64]
  ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    return proj.T @ r, proj.T @ proj


def minimize_api() -> None:
  def residual(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return x - 1.0

  def jac(
    x: npt.NDArray[np.float64], r: npt.NDArray[np.float64]
  ) -> npt.NDArray[np.float64]:
    return np.eye(x.size)

  def on_iter(trace: list[minimize.IterLog]) -> None:
    assert_type(trace[-1].objective, float)
    assert_type(trace[-1].candidate, npt.NDArray[np.float64])
    assert_type(trace[-1].residual, npt.NDArray[np.float64] | None)

  x0 = np.zeros(2)
  x, trace = minimize.least_squares(
    x0,
    residual,
    bounds=[np.full(2, -2.0), np.full(2, 2.0)],
    jacobian=jac,
    norm=Squared(),
    verbose=minimize.Verbosity.FULLITER,
    iter_callback=on_iter,
    x_scale='jac',
  )
  assert_type(x, npt.NDArray[np.float64])
  assert_type(trace, list[minimize.IterLog])
  assert_type(trace[0].regularizer, float)
  minimize.least_squares(np.array([1, 2]), residual, verbose=0, x_scale=2.0)
  assert_type(minimize.Quadratic().value(x), float)
  grad, hess = minimize.Quadratic().grad_hess(x, np.eye(2))
  assert_type(grad, npt.NDArray[np.float64])
  assert_type(hess, npt.NDArray[np.float64])
  fd, n_res = minimize.jacobian_fd(residual, x, x, 1e-8, 0)
  assert_type(fd, npt.NDArray[np.float64])
  assert_type(n_res, int)
  assert_type(minimize.check_jacobian(residual, x, x, fd, 1e-8, 0), int)
  assert_type(minimize.check_norm(x, Squared(), 1e-8), None)
  assert_type(minimize.Verbosity(2), minimize.Verbosity)
  assert_type(minimize.Status(5), minimize.Status)
  log = minimize.IterLog(x, 0.0, 0.0, 0.0)
  assert_type(log.step, npt.NDArray[np.float64] | None)


def msh2obj_api() -> None:
  msh = msh2obj.Msh.create(pathlib.Path('mesh.msh'))
  assert_type(msh, msh2obj.Msh)
  assert_type(msh.vertex_positions, npt.NDArray[np.float32])
  assert_type(msh.face_vertex_indices, npt.NDArray[np.int32])
  assert_type(msh2obj.msh_to_obj(pathlib.Path('mesh.msh')), str)


def usd_helpers() -> None:
  matrix = utils.create_transform_matrix(np.eye(3), [0.0, 0.0, 1.0])
  assert_type(matrix, npt.NDArray[np.float64])
  assert_type(mujoco._usd.register_plugins(), None)
  assert_type(mujoco._usd.__version__, str)

  box = shapes.TriangleMesh.create_box(1.0, 2.0, 3.0, None)
  assert_type(box, shapes.TriangleMesh)
  assert_type(box.vertices, npt.NDArray[np.float64])
  assert_type(box.triangle_uvs, npt.NDArray[np.float64] | None)
  assert_type(box.get_center(), npt.NDArray[np.float64])
  box.translate((0.0, 0.0, 1.0))
  assert_type(box + box, shapes.TriangleMesh)
  tex = mujoco.mjtTexture.mjTEXTURE_2D
  assert_type(
    shapes.get_triangle_uvs(box.vertices, box.triangles, tex),
    npt.NDArray[np.float64],
  )
  assert_type(shapes.get_triangle_uvs(box.vertices, box.triangles, None), None)
  size = np.array([1.0, 1.0, 1.0])
  geom_type = mujoco.mjtGeom.mjGEOM_BOX
  assert_type(
    shapes.mesh_config_generator('box', geom_type, size), dict[str, Any]
  )
  parts = shapes.mesh_config_generator('box', geom_type, size, decouple=True)
  assert_type(parts, list[dict[str, Any]])
  name, mesh = shapes.mesh_factory(parts[0], tex, 10)
  assert_type(name, str)
  assert_type(mesh, shapes.TriangleMesh)
  assert_type(shapes.decouple_config({'name': 'x'}), list[dict[str, Any]])


def usd_exporter(model: mujoco.MjModel, data: mujoco.MjData) -> None:
  exp = exporter.USDExporter(model, height=64, width=64, camera_names=['cam'])
  assert_type(exp, exporter.USDExporter)
  assert_type(exp.usd, str)
  assert_type(exp.scene, mujoco.MjvScene)
  assert_type(exp.frame_count, int)
  assert_type(exp.camera_names, list[str] | None)
  assert_type(exp.geom_refs, dict[str, objects.USDObject])
  assert_type(exp.usd_lights, list[lights.USDSphereLight | None])
  assert_type(exp.usd_cameras, list[camera.USDCamera])
  exp.update_scene(data, mujoco.MjvOption())
  exp.add_light(pos=(0.0, 0.0, 0.0), intensity=2000, light_type='dome')
  exp.add_camera([0.0, 0.0, 1.0], [0.0, 0.0, 0.0])
  exp.save_scene(filetype='usda')
  assert_type(mujoco.usd.exporter.USDExporter(model), exporter.USDExporter)
  local_args = argparse.Namespace()
  assert_type(demo.generate_usd_trajectory(local_args), None)


def usd_objects(
  stage: object, model: mujoco.MjModel, geom: mujoco.MjvGeom
) -> None:
  prim = objects.USDPrimitiveMesh(
    shapes.mesh_config_generator('s', geom.type, geom.size),
    stage,
    model,
    geom,
    's',
  )
  assert_type(prim.prim_mesh, shapes.TriangleMesh)
  assert_type(prim.obj_name, str)
  assert_type(prim.last_visible_frame, int)
  assert_type(prim.generate_primitive_mesh(), shapes.TriangleMesh)
  prim.update(geom.pos, geom.mat, True, 0)
  tendon = objects.USDTendon(
    shapes.mesh_config_generator('t', geom.type, geom.size, decouple=True),
    stage,
    model,
    geom,
    't',
  )
  assert_type(tendon.tendon_parts, dict[str, shapes.TriangleMesh])
  mesh = objects.USDMesh(stage, model, geom, 'm', 0)
  assert_type(mesh.dataid, int)
  light = lights.USDSphereLight(stage, 'l', radius=0.5)
  light.update(geom.pos, 10, geom.rgba[:3], 0)
  lights.USDDomeLight(stage, 'd').update(5, geom.rgba[:3])
  cam = camera.USDCamera(stage, 'c')
  cam.update(geom.pos, geom.mat, 0)


def usd_native(
  stage: object, spec: mujoco.MjSpec, model: mujoco.MjModel, data: mujoco.MjData
) -> None:
  usd_writer = writer.UsdWriter(stage, spec, model)
  sink = layer_sink.LayerSink(stage)
  usd_writer.add_sink(sink.update)
  assert_type(usd_writer.update(data), None)


def usd_schemas(stage: object, prim: object) -> None:
  collision = MjcPhysics.CollisionAPI.Apply(prim)
  assert_type(collision, MjcPhysics.CollisionAPI)
  collision.CreateGroupAttr().Set(3)
  assert_type(MjcPhysics.CollisionAPI.CanApply(prim).whyNot, str)
  assert_type(bool(MjcPhysics.CollisionAPI.CanApply(prim)), bool)
  actuator = MjcPhysics.Actuator.Define(stage, '/actuator')
  assert_type(actuator, MjcPhysics.Actuator)
  assert_type(MjcPhysics.Actuator.GetSchemaAttributeNames(), list[str])
  assert_type(MjcPhysics.Tokens.mjcAct, str)
