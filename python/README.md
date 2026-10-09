# MuJoCo Python Bindings

[![PyPI Python Version][pypi-versions-badge]][pypi]
[![PyPI version][pypi-badge]][pypi]

[pypi-versions-badge]: https://img.shields.io/pypi/pyversions/mujoco
[pypi-badge]: https://badge.fury.io/py/mujoco.svg
[pypi]: https://pypi.org/project/mujoco/

This package is the canonical Python bindings for the
[MuJoCo physics engine](https://github.com/google-deepmind/mujoco).
These bindings are developed and maintained by Google DeepMind, and is kept
up-to-date with the latest developments in MuJoCo itself.

The `mujoco` package provides direct access to raw MuJoCo C API functions,
structs, constants, and enumerations. Structs are provided as Python classes,
with Pythonic initialization and deletion semantics.

It is not the aim of this package to provide fully fledged
scene/environment/game authoring API, as there are already a number of existing
packages that do this well. However, this package does provide a number of
lower-level components outside of MuJoCo itself that are likely to be useful to
most users who access MuJoCo through Python. For example, the `egl`, `glfw`, and
`osmesa` subpackages contain utilities for setting up OpenGL rendering contexts.

## Installation

The recommended way to install this package is via [PyPI](https://pypi.org/project/mujoco/):

```sh
pip install mujoco
```

A copy of the MuJoCo library is provided as part of the package and does **not**
need to be downloaded or installed separately.

### Source

**IMPORTANT:** Building from source is only necessary if you are modifying the
Python bindings (or are trying to run on exceptionally old Linux systems).
If that's not the case, then we recommend installing the prebuilt binaries from
PyPI.

If you need to build the Python bindings from source, please consult
[the documentation](https://mujoco.readthedocs.io/en/latest/python.html#building-from-source).

## Type hints

Wheels built from this checkout bundle `.pyi` files for ten core native
modules and public Python wrappers, plus `py.typed`. From the repository root,
install a matching built wheel in an active virtual environment (use one wheel
file if `python/dist` contains several):

```sh
uv pip install python/dist/mujoco-3.15.1-*.whl
```

The installed wheel needs no separate stub package or `stubPath` setting.
Existing PyPI releases do not yet include these changes. The root `py.typed`
marker does not mean every optional module is fully typed: Studio and Filament
native extensions have no new stubs, and the separate `mujoco-mjx` distribution
is outside this work. The hints do not specify exact NumPy array shapes or the
compile-time precision of `mjtNum`; `MJTNUM_DTYPE` depends on the build.

To regenerate the native stubs, use Python 3.11 or newer (`tomllib` is required)
and a matching double-precision MuJoCo 3.15.1 wheel built from this checkout.
From the repository root, with that wheel installed in the active virtual
environment, run:

```sh
uv pip install --group python/pyproject.toml:typing
python python/mujoco/codegen/generate_stubs.py --output python/mujoco
python python/tests/typing/check.py
```

Review the generated stubs before committing them. The generator and typing
tests are developer tools in the repository, not part of the sdist or wheel.
Wheel builds package the checked-in stubs without regenerating them. A source
install such as `uv pip install ./python` still needs a native MuJoCo SDK,
CMake, `MUJOCO_PATH`, and `MUJOCO_PLUGIN_PATH`. The repository root is not an
installable Python project.

## Usage

Once installed, the package can be imported via `import mujoco`. Please consult
our [documentation](https://mujoco.readthedocs.io/en/stable/python.html) for
further detail on the package's API.

We recommend going through the tutorial notebook which covers the basics of
MuJoCo using Python:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/python/tutorial.ipynb)

## Versioning

The `major.minor.micro` portion of the version number matches the version of
MuJoCo that the bindings provide. Optionally, if we release updates to the
Python bindings themselves that target the same version of MuJoCo, a `.postN`
suffix is added, for example `2.1.2.post2` represents the second update to the
bindings for MuJoCo 2.1.2.

## License and Disclaimer

Copyright 2022 DeepMind Technologies Limited

MuJoCo and its Python bindings are licensed under the Apache License,
Version 2.0. You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0.

This is not an officially supported Google product.
