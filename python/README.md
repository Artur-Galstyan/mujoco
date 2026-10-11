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

### Install from Git

To build this fork from source on Python 3.13, run:

```sh
uv pip install 'mujoco @ git+https://github.com/Artur-Galstyan/mujoco.git'
```

The build compiles the MuJoCo engine and its four bundled plugins from the
same full checkout, then builds the classic Python bindings. You do not need a
separate MuJoCo SDK or `MUJOCO_PATH` / `MUJOCO_PLUGIN_PATH`. You do need a C++
compiler, Git, and network access for pinned CMake dependencies. Build isolation
installs CMake and Python build dependencies, including `absl-py`. The first
build can take several minutes. Optional Studio and Filament are not part of
this source build.

For a local full checkout, run `uv pip install .` from the repository root
(`uv pip install ./python` still works).

### Source archives and external SDKs

A standalone Python source archive does not include the full engine checkout.
To build from such an archive, or to use an existing MuJoCo release instead,
set `MUJOCO_PATH` to the matching SDK and `MUJOCO_PLUGIN_PATH` to its built
plugins. See the [source build documentation](https://mujoco.readthedocs.io/en/latest/python.html#building-from-source)
for details. The Git and local full-checkout commands above need neither path.

## Type hints

For the latest official MuJoCo release (3.15.0), use the separate
[`typed-mujoco`](../typed-mujoco/README.md) companion package. From the
repository root, install it alongside the matching official runtime:

```sh
uv pip install 'mujoco==3.15.0' ./typed-mujoco
```

The companion provides `mujoco-stubs` for static type checking; it does not
replace the runtime. The 3.15.1 development-fork wheels built from this
checkout do not bundle core stubs or a root `py.typed` marker, and the
companion does not type-check this fork. The `mujoco.sysid` subpackage keeps
its own typing marker. MJX (`mujoco-mjx`) has separate types and tests.

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
