# typed-mujoco

`typed-mujoco` is an **unofficial** companion of the official `mujoco` Python
package. It ships 32 partial PEP 561 stubs, not the engine or a runtime fork.
These stubs target the official MuJoCo 3.17.0 wheel.
The Apache 2.0 license and upstream copyright notices remain with the stubs.

Version `3.17.0.0` requires `mujoco>=3.17.0` and Python 3.10 or newer,
matching the Python floor published for the official 3.17.0 wheel. Later
releases are allowed, but need a compatibility check against their installed
API. Update the stubs if that check finds changes.
The `partial` marker identifies modules without companion stubs. However,
`ty==0.0.85` does not resolve upstream-only modules such as `mujoco.minimize`;
it reports `unresolved-import`. Other uncovered modules and MJX, if installed,
may fail static imports too, even though runtime imports work. The checks verify
this Ty error against a real upstream module; MJX fallback is not covered.
Optional Filament and Studio native modules are not supplied by this companion
and may be unavailable at runtime. Their stubs always receive static checks;
the native API audit reports which optional modules are absent.

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

From the repository root, build and check the wheel against the 3.17.0 baseline
with Python 3.13:

```sh
uv venv --python 3.13 typed-mujoco/.venv
uv build typed-mujoco --python 3.13 --out-dir typed-mujoco/dist
uv pip install --python typed-mujoco/.venv/bin/python 'mujoco==3.17.0'
uv pip install --python typed-mujoco/.venv/bin/python typed-mujoco/dist/typed_mujoco-3.17.0.0-py3-none-any.whl
uv pip install --python typed-mujoco/.venv/bin/python --group typed-mujoco/pyproject.toml:typing
PATH="/usr/sbin:typed-mujoco/.venv/bin:$PATH" MUJOCO_GL=disable typed-mujoco/.venv/bin/python -I typed-mujoco/tests/check.py
typed-mujoco/.venv/bin/ruff check --config typed-mujoco/pyproject.toml typed-mujoco/tests typed-mujoco/mujoco-stubs
typed-mujoco/.venv/bin/ruff format --check --config typed-mujoco/pyproject.toml typed-mujoco/tests typed-mujoco/mujoco-stubs
uv build typed-mujoco/dist/typed_mujoco-3.17.0.0.tar.gz --wheel --python 3.13 --out-dir typed-mujoco/dist/roundtrip
uv pip install --python typed-mujoco/.venv/bin/python typed-mujoco/dist/roundtrip/typed_mujoco-3.17.0.0-py3-none-any.whl
PATH="/usr/sbin:typed-mujoco/.venv/bin:$PATH" MUJOCO_GL=disable typed-mujoco/.venv/bin/python -I typed-mujoco/tests/check.py
```

To check the latest official release locally, upgrade MuJoCo and run the same
checker again:

```sh
uv pip install --python typed-mujoco/.venv/bin/python --upgrade --default-index https://pypi.org/simple mujoco
PATH="/usr/sbin:typed-mujoco/.venv/bin:$PATH" MUJOCO_GL=disable typed-mujoco/.venv/bin/python -I typed-mujoco/tests/check.py
```

The checker tests the installed wheel's file list, the MuJoCo API, documented
examples, runtime smoke tests, Ty consumers and negative cases, and Ruff. It
checks the sdist-derived wheel too. It reports optional native modules that the
runtime wheel does not include. CI runs independent baseline (3.17.0) and latest
MuJoCo checks on macOS and Ubuntu, on scoped pushes and pull requests, manual
runs, and daily at 06:17 UTC. Each CI run checks that MuJoCo still imports after
the companion is uninstalled.
