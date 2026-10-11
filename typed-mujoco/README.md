# typed-mujoco

`typed-mujoco` is an **unofficial** companion of the official `mujoco` Python
package. It ships 29 partial PEP 561 stubs, not the engine or a runtime fork.
These stubs are manually maintained against the official MuJoCo 3.15.0 wheel.
The Apache 2.0 license and upstream copyright notices remain with the stubs.

Version `3.15.0.0` requires `mujoco==3.15.0` and Python 3.10 or newer.
Only the latest official release is supported: a new MuJoCo release needs
another audit of its installed API and an updated companion version and stubs.
The `partial` marker identifies modules without companion stubs. However,
`ty==0.0.85` does not resolve upstream-only modules such as `mujoco.minimize`;
it reports `unresolved-import`. Other uncovered modules and MJX, if installed,
may fail static imports too, even though runtime imports work. The tests check
the ty error on a real upstream module; MJX fallback is not covered. Optional
Filament and Studio native modules are not supplied by this companion and
may be unavailable at runtime.

No publication was done. From the repository root, build and check the
installed wheel with Python 3.13:

```sh
uv venv --python 3.13 typed-mujoco/.venv
uv build typed-mujoco --python 3.13 --out-dir typed-mujoco/dist
uv pip install --python typed-mujoco/.venv/bin/python typed-mujoco/dist/typed_mujoco-3.15.0.0-py3-none-any.whl
uv pip install --python typed-mujoco/.venv/bin/python --group typed-mujoco/pyproject.toml:typing
PATH="/usr/sbin:typed-mujoco/.venv/bin:$PATH" MUJOCO_GL=disable typed-mujoco/.venv/bin/python -I typed-mujoco/tests/check.py
typed-mujoco/.venv/bin/ruff check --config typed-mujoco/pyproject.toml typed-mujoco/tests typed-mujoco/mujoco-stubs
typed-mujoco/.venv/bin/ruff format --check --config typed-mujoco/pyproject.toml typed-mujoco/tests typed-mujoco/mujoco-stubs
uv build typed-mujoco/dist/typed_mujoco-3.15.0.0.tar.gz --wheel --python 3.13 --out-dir typed-mujoco/dist/roundtrip
uv pip install --python typed-mujoco/.venv/bin/python typed-mujoco/dist/roundtrip/typed_mujoco-3.15.0.0-py3-none-any.whl
PATH="/usr/sbin:typed-mujoco/.venv/bin:$PATH" MUJOCO_GL=disable typed-mujoco/.venv/bin/python -I typed-mujoco/tests/check.py
```

The CI workflow also checks that MuJoCo 3.15.0 still imports after the
companion is uninstalled. Release owners must audit the matching official
runtime again before any future publication.
