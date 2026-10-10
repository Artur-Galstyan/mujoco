"""Type-check only: each marked line must produce a ty error."""

import jax
import numpy as np
from typing_extensions import assert_type

import mujoco
from mujoco import mjx


def check_bad_calls(
  model: mjx.Model, data: mjx.Data, host: mujoco.MjModel, array: jax.Array
) -> None:
  mjx.put_model('not a model')  # ty-error
  mjx.make_data('not a model')  # ty-error
  mjx.set_state(
    model,
    data,
    np.array([0.0]),  # ty-error
    mujoco.mjtState.mjSTATE_QPOS,
  )
  mjx.get_data_into('not data', host, data)  # ty-error
  mjx.render(model, data, object())  # ty-error
  mjx.refit_bvh(model, data, object())  # ty-error
  data.tree_replace({'qpos': object()})  # ty-error
  assert_type(data.replace(qpos=array), int)  # ty-error

  mjx.Contact(
    dist='not a JAX array',  # ty-error
    pos=array,
    frame=array,
    includemargin=array,
    friction=array,
    solref=array,
    solreffriction=array,
    solimp=array,
    dim=array,  # ty-error
    geom1=array,
    geom2=array,
    geom=array,
    efc_address=np.array([0]),
  )
