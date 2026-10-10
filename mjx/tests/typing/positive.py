"""Type-check only: examples of the installed MJX API."""

import jax
import numpy as np
from typing_extensions import assert_type

import mujoco
from mujoco import mjx
from mujoco.mjx.warp.render_context import RenderContext, RenderContextPytree
from mujoco.mjx.warp.types import ModelWarp


def check_public_api(
  model: mjx.Model,
  data: mjx.Data,
  host: mujoco.MjModel,
  host_data: mujoco.MjData,
  array: jax.Array,
  static: np.ndarray,
  context: RenderContextPytree,
) -> None:
  assert_type(model.body_parentid, np.ndarray)
  assert_type(model._impl, mjx.ModelCPP | mjx.ModelJAX | ModelWarp)
  assert_type(data.qpos, jax.Array)
  assert_type(mjx.put_model(host, impl='jax'), mjx.Model)
  assert_type(mjx.make_data(model, impl='jax'), mjx.Data)
  assert_type(mjx.put_data(host, host_data, impl='jax'), mjx.Data)
  assert_type(mjx.step(model, data), mjx.Data)
  assert_type(mjx.forward(model, data), mjx.Data)
  assert_type(data.replace(qpos=array), mjx.Data)
  assert_type(data.tree_replace({'qpos': array}), mjx.Data)
  assert_type(data[0], mjx.Data)
  assert_type(mjx.get_data_into(host_data, host, data), None)
  assert_type(mjx.get_data(host, data), mujoco.MjData | list[mujoco.MjData])
  assert_type(mjx.create_render_context(host, 1), RenderContext)
  assert_type(mjx.create_render_context(host, 1).pytree(), RenderContextPytree)
  assert_type(
    mjx.render(model, data, context), tuple[jax.Array, jax.Array, mjx.Data]
  )
  assert_type(mjx.refit_bvh(model, data, context), mjx.Data)
  assert_type(
    mjx.get_state(model, data, mujoco.mjtState.mjSTATE_QPOS), jax.Array
  )

  contact = mjx.Contact(
    dist=array,
    pos=array,
    frame=array,
    includemargin=array,
    friction=array,
    solref=array,
    solreffriction=array,
    solimp=array,
    dim=static,
    geom1=array,
    geom2=array,
    geom=array,
    efc_address=static,
  )
  assert_type(contact, mjx.Contact)
  assert_type(contact.dim, np.ndarray)
