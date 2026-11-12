# typed-mujoco

`typed-mujoco` is an **unofficial** companion of the official `mujoco` Python
package. It ships 117 PEP 561 stubs, not the engine or a runtime fork.
These stubs target the official MuJoCo 4.0.1 wheel.
The Apache 2.0 license and upstream copyright notices remain with the stubs.

Version `4.0.1.1` requires `mujoco>=4.0.1` and Python 3.11 or newer,
matching the Python floor published for the official 4.0.1 wheel. Later
releases are allowed, but need a compatibility check against their installed
API. Update the stubs if that check finds changes.
The stubs cover every non-test Python module of the official `mujoco`,
`mujoco-core`, `mujoco-studio`, and `mujoco-usd` 4.0.1 wheels: the native
modules and the pure-Python modules, such as `mujoco.minimize`, `mujoco.sysid`,
`mujoco.usd`, `mujoco.introspect`, and `mujoco.studio`. The checks fail when an
upstream module has no stub.
Some stubs import optional third-party packages, such as `scipy`, `plotly`, and
`pxr`. Install the matching MuJoCo extra (`mujoco[sysid]`, `mujoco[usd]`) to
type-check code that uses those modules. `pxr` and `PyOpenGL` ship no types, so
their classes resolve to `Any`.
`ty==0.0.85` does not read upstream sources for modules without a stub. MJX is
a separate distribution without companion stubs here, so `mujoco.mjx` imports
report `unresolved-import` under Ty, even though runtime imports work.
Optional Filament, Studio, and USD native modules are not supplied by this
companion and may be unavailable at runtime. Their stubs always receive static
checks; the native API audit reports which optional modules are absent.

## Installation

Install the companion with:

```sh
uv pip install typed-mujoco
```

For an existing installation, upgrade both packages with:

```sh
uv pip install --upgrade typed-mujoco mujoco
```

```python
import mujoco

xml = (
    '<mujoco><worldbody><body name="ball"><freejoint/>'
    '<geom size=".1"/></body></worldbody></mujoco>'
)
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
```

## Developer build and checks

From the repository root, build and check the wheel against the 4.0.1 baseline
with Python 3.13:

```sh
uv venv --python 3.13 typed-mujoco/.venv
uv build typed-mujoco --python 3.13 --out-dir typed-mujoco/dist
uv pip install --python typed-mujoco/.venv/bin/python 'mujoco[sysid,usd]==4.0.1'
uv pip install --python typed-mujoco/.venv/bin/python typed-mujoco/dist/typed_mujoco-4.0.1.1-py3-none-any.whl
uv pip install --python typed-mujoco/.venv/bin/python --group typed-mujoco/pyproject.toml:typing
PATH="/usr/sbin:typed-mujoco/.venv/bin:$PATH" MUJOCO_GL=disable typed-mujoco/.venv/bin/python -I typed-mujoco/tests/check.py
typed-mujoco/.venv/bin/ruff check --config typed-mujoco/pyproject.toml typed-mujoco/tests typed-mujoco/mujoco-stubs
typed-mujoco/.venv/bin/ruff format --check --config typed-mujoco/pyproject.toml typed-mujoco/tests typed-mujoco/mujoco-stubs
uv build typed-mujoco/dist/typed_mujoco-4.0.1.1.tar.gz --wheel --python 3.13 --out-dir typed-mujoco/dist/roundtrip
uv pip install --python typed-mujoco/.venv/bin/python typed-mujoco/dist/roundtrip/typed_mujoco-4.0.1.1-py3-none-any.whl
PATH="/usr/sbin:typed-mujoco/.venv/bin:$PATH" MUJOCO_GL=disable typed-mujoco/.venv/bin/python -I typed-mujoco/tests/check.py
```

To check the latest official release locally, upgrade MuJoCo and run the same
checker again:

```sh
uv pip install --python typed-mujoco/.venv/bin/python --upgrade --default-index https://pypi.org/simple 'mujoco[sysid,usd]'
PATH="/usr/sbin:typed-mujoco/.venv/bin:$PATH" MUJOCO_GL=disable typed-mujoco/.venv/bin/python -I typed-mujoco/tests/check.py
```

The checker tests the installed wheel's file list, the MuJoCo API, documented
examples, runtime smoke tests, Ty consumers and negative cases, and Ruff. It
checks the sdist-derived wheel too. It reports optional native modules that the
runtime wheel does not include. CI runs independent baseline (4.0.1) and latest
MuJoCo checks on macOS and Ubuntu, on scoped pushes and pull requests, and manual
runs. Each CI run checks that MuJoCo still imports after
the companion is uninstalled.
