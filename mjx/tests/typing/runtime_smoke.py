"""CPU smoke for an installed MJX distribution and its PyTree metadata."""

import dataclasses
import tomllib
from importlib import metadata, resources
from pathlib import Path
from typing import get_args

import jax
import numpy as np

import mujoco
from mujoco import mjx
from mujoco.mjx._src.dataclasses import _jax_in_args
from mujoco.mjx.warp.types import ModelWarp

assert resources.files('mujoco.mjx').joinpath('py.typed').is_file()
distribution = metadata.distribution('mujoco-mjx')
assert (
  Path(mjx.__file__).resolve()
  == Path(distribution.locate_file('mujoco/mjx/__init__.py')).resolve()
)
with (Path(__file__).resolve().parents[2] / 'pyproject.toml').open('rb') as f:
  expected_version = tomllib.load(f)['project']['version']
assert distribution.version == expected_version

host = mujoco.MjModel.from_xml_string(
  '<mujoco><worldbody><body><joint type="hinge"/>'
  '<geom type="sphere" size=".1"/></body></worldbody></mujoco>'
)
cpu = jax.devices('cpu')[0]
model = mjx.put_model(host, device=cpu, impl='jax')
data = mjx.make_data(model, device=cpu, impl='jax')
assert isinstance(model.body_parentid, np.ndarray)
assert isinstance(data.qpos, jax.Array)

# The CPP type in the Union must not turn the implementation into static metadata.
impl_type = next(
  f.type for f in dataclasses.fields(mjx.Model) if f.name == '_impl'
)
assert get_args(impl_type) == (mjx.ModelCPP, mjx.ModelJAX, ModelWarp)
assert _jax_in_args(impl_type)

leaves, tree = jax.tree_util.tree_flatten(model)
reconstructed = tree.unflatten(leaves)
assert isinstance(reconstructed._impl, mjx.ModelJAX)
assert reconstructed.body_parentid is model.body_parentid
assert all(leaf is not model.body_parentid for leaf in leaves)

cpp = mjx.put_model(host, device=cpu, impl='cpp', keepalive_refs={})
cpp_leaves, cpp_tree = jax.tree_util.tree_flatten_with_path(cpp)
assert any(
  len(path) >= 2 and path[0].name == '_impl' and path[1].name == 'pointer_lo'
  for path, _ in cpp_leaves
)
assert isinstance(
  cpp_tree.unflatten([leaf for _, leaf in cpp_leaves])._impl, mjx.ModelCPP
)

updated = data.replace(qpos=data.qpos + 0.1)
nested = data.tree_replace({'qpos': updated.qpos})
np.testing.assert_allclose(nested.qpos, updated.qpos)
assert data.qpos[0] != updated.qpos[0]
stepped = mjx.step(model, nested)
assert isinstance(stepped, mjx.Data)
assert isinstance(mjx.get_data(host, stepped), mujoco.MjData)
assert mjx.get_data_into(mujoco.MjData(host), host, stepped) is None
print('installed MJX CPU PyTree and step: OK')
