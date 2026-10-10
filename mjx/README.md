# MuJoCo XLA (MJX)

[![PyPI Python Version][pypi-versions-badge]][pypi]
[![PyPI version][pypi-badge]][pypi]

[pypi-versions-badge]: https://img.shields.io/pypi/pyversions/mujoco-mjx
[pypi-badge]: https://badge.fury.io/py/mujoco-mjx.svg
[pypi]: https://pypi.org/project/mujoco-mjx/

This package is a re-implementation of the
[MuJoCo physics engine](https://github.com/google-deepmind/mujoco) in
[JAX](https://github.com/jax-ml/jax). This library is developed and maintained
by Google DeepMind, and is kept up-to-date with the latest developments in
MuJoCo itself.

The `mujoco-mjx` package is API-compatible with MuJoCo, but is missing some
features found in MuJoCo.  See our
[documentation](https://mujoco.readthedocs.io/en/stable/mjx.html) for more
details concerning feature parity.

## Installation

The recommended way to install this package is via [PyPI](https://pypi.org/project/mujoco-mjx/):

```sh
pip install mujoco-mjx
```

## Usage

Once installed, the package can be imported via `from mujoco import mjx`. Please
consult our [documentation](https://mujoco.readthedocs.io/en/stable/mjx.html)
for further detail on the package's API.

We recommend going through the tutorial notebook which introduces the MJX API
and trains a reinforcement learning policy in a few minutes: [![Open In
Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/google-deepmind/mujoco/blob/main/mjx/tutorial.ipynb)

## Typing verification

Run these commands from the repository root. The local MuJoCo install
builds the core from this checkout and needs a C++ compiler and network access
for pinned CMake dependencies. The first build can take several minutes.
The separate environment does not use other installed-consumer directories.

```sh
uv venv --python 3.13 build/mjx-typing-consumer/venv
uv build mjx --out-dir build/mjx-typing-consumer/dist --python build/mjx-typing-consumer/venv/bin/python
uv pip install --python build/mjx-typing-consumer/venv/bin/python .
uv pip install --python build/mjx-typing-consumer/venv/bin/python build/mjx-typing-consumer/dist/mujoco_mjx-3.15.1-py3-none-any.whl ty==0.0.85 ruff==0.17.0
build/mjx-typing-consumer/venv/bin/python mjx/tests/typing/check.py
build/mjx-typing-consumer/venv/bin/python mjx/tests/typing/runtime_smoke.py
build/mjx-typing-consumer/venv/bin/python -m unittest mujoco.mjx._src.dataclasses_test mujoco.mjx._src.io_test.DataIOTest.test_get_data_runs
build/mjx-typing-consumer/venv/bin/ruff check --isolated --select E4,E7,E9,F mjx/tests/typing
```

The type check copies the consumer examples to a temporary project. It ignores
user ty configuration and checks installed package paths. The CPU smoke test
checks `py.typed`, JAX and CPP PyTree metadata, and a JAX step. Warp rendering
needs a separate Warp installation and is not part of these CPU checks.

## Versioning

The `major.minor.micro` portion of the version number matches the version of
MuJoCo that this library provides. Optionally, if we release updates to MJX that
target the same version of MuJoCo, a `.postN` suffix is added, for example
`3.0.1.post2` represents the second update to MJX for MuJoCo 3.0.1.

## License and Disclaimer

Copyright 2023 DeepMind Technologies Limited

MuJoCo and its libraries are licensed under the Apache License,
Version 2.0. You may obtain a copy of the License at
https://www.apache.org/licenses/LICENSE-2.0.

This is not an officially supported Google product.
