# Copyright 2026 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for source builds without an external MuJoCo SDK."""

import os
import platform
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import setuptools
import tomllib

PYTHON = Path(__file__).resolve().parents[1]
CORE = (
  '_callbacks',
  '_constants',
  '_enums',
  '_errors',
  '_functions',
  '_render',
  '_rollout',
  '_simulate',
  '_specs',
  '_structs',
)
PLUGINS = ('actuator', 'elasticity', 'sensor', 'sdf_plugin')


class RepositoryInstallTest(unittest.TestCase):
  def test_root_build_requirements_match_python_package(self):
    root = tomllib.loads((PYTHON.parent / 'pyproject.toml').read_text())
    child = tomllib.loads((PYTHON / 'pyproject.toml').read_text())
    self.assertEqual(root['build-system'], child['build-system'])

  def test_root_uses_child_setup_and_restores_working_directory(self):
    calls = []
    with tempfile.TemporaryDirectory() as directory:
      temp = Path(directory)
      if os.name != 'nt':
        temp = temp / 'alias'
        temp.symlink_to(directory, target_is_directory=True)
      previous = Path.cwd()
      try:
        os.chdir(temp)
        with mock.patch.object(
          setuptools,
          'setup',
          side_effect=lambda **kwargs: calls.append(
            (Path.cwd(), Path(sys.argv[0]))
          ),
        ) as setup:
          runpy.run_path(str(PYTHON.parent / 'setup.py'))
        self.assertEqual(Path.cwd(), Path(temp).resolve())
      finally:
        os.chdir(previous)
    setup.assert_called_once()
    self.assertEqual(calls, [(PYTHON, PYTHON / 'setup.py')])
    self.assertEqual(
      setup.call_args.kwargs['cmdclass'][
        'build_ext'
      ]._prepare_source.__globals__['__file__'],
      str(PYTHON / 'setup.py'),
    )
    self.assertTrue(
      setup.call_args.kwargs['long_description'].startswith(
        (PYTHON / 'README.md').read_text()
      )
    )

  def test_root_restores_working_directory_when_setup_fails(self):
    with tempfile.TemporaryDirectory() as temp:
      previous = Path.cwd()
      try:
        os.chdir(temp)
        with mock.patch.object(
          setuptools, 'setup', side_effect=RuntimeError('failed')
        ):
          with self.assertRaisesRegex(RuntimeError, 'failed'):
            runpy.run_path(str(PYTHON.parent / 'setup.py'))
        self.assertEqual(Path.cwd(), Path(temp).resolve())
      finally:
        os.chdir(previous)


class SourceBuildTest(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    previous = Path.cwd()
    try:
      os.chdir(PYTHON)
      with mock.patch.object(setuptools, 'setup') as setup:
        cls.module = runpy.run_path(str(PYTHON / 'setup.py'))
      setup.assert_called_once()
      cls.extensions = setup.call_args.kwargs['ext_modules']
      cls.command_class = setup.call_args.kwargs['cmdclass']['build_ext']
    finally:
      os.chdir(previous)

  def setUp(self):
    tmp = tempfile.TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    self.tmp = Path(tmp.name)
    self.command = self.command_class(setuptools.Distribution())
    self.command.build_temp = str(self.tmp / 'build')
    self.command.build_lib = str(self.tmp / 'wheel')
    self.command.extensions = self.extensions
    self.command.debug = False
    self.command._is_apple = platform.system() == 'Darwin'

  def sdk(self):
    root = self.tmp / 'sdk'
    header = root / 'include' / 'mujoco' / 'mujoco.h'
    header.parent.mkdir(parents=True)
    header.touch()
    library = {
      'Windows': 'mujoco.lib',
      'Darwin': 'libmujoco.3.15.1.dylib',
    }.get(platform.system(), 'libmujoco.so.3.15.1')
    lib = root / 'lib'
    lib.mkdir()
    (lib / library).touch()
    return root, lib, header.parent.parent

  def fake_source(self, prepared=False):
    root = self.tmp / 'checkout'
    source = root / 'python' / 'mujoco'
    source.mkdir(parents=True)
    if prepared:
      names = (
        'enum_traits.h',
        'function_traits.h',
        'specs.cc.inc',
        'cmake/MujocoOptions.cmake',
        'simulate/CMakeLists.txt',
      )
    else:
      names = (
        'codegen/generate_enum_traits.py',
        'codegen/generate_function_traits.py',
        'codegen/generate_spec_bindings.py',
      )
      for name in (
        'CMakeLists.txt',
        'include/mujoco/mujoco.h',
        'cmake/MujocoOptions.cmake',
        'simulate/CMakeLists.txt',
        'plugin/sdf/CMakeLists.txt',
      ):
        file = root / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(name)
    for name in names:
      file = source / name
      file.parent.mkdir(parents=True, exist_ok=True)
      file.write_text(name)
    return root, source

  def test_checkout_without_sdk_builds_engine(self):
    sdk, lib, include = self.sdk()
    self.command._repo_root = str(self.tmp / 'checkout')

    def build_engine():
      self.command._sdk_root = str(sdk)

    with mock.patch.dict(os.environ, {}, clear=True):
      with mock.patch.object(
        self.command, '_build_mujoco', side_effect=build_engine
      ) as build:
        paths = self.command._find_mujoco()
    build.assert_called_once_with()
    self.assertEqual(
      paths, (str(lib), str(include), str(sdk / 'mujoco_plugin'), None)
    )

  def test_incomplete_sdk_environment_does_not_build_engine(self):
    self.command._repo_root = str(self.tmp / 'checkout')
    for env in (
      {'MUJOCO_PATH': 'sdk'},
      {'MUJOCO_PLUGIN_PATH': 'plugins'},
      {'MUJOCO_PATH': '', 'MUJOCO_PLUGIN_PATH': 'plugins'},
    ):
      with self.subTest(env=env), mock.patch.dict(os.environ, env, clear=True):
        with mock.patch.object(self.command, '_build_mujoco') as build:
          with self.assertRaisesRegex(RuntimeError, 'both MUJOCO_PATH'):
            self.command._find_mujoco()
        build.assert_not_called()

  def test_explicit_sdk_is_used_without_rebuilding(self):
    sdk, lib, include = self.sdk()
    plugins = self.tmp / 'external_plugins'
    plugins.mkdir()
    self.command._repo_root = str(self.tmp / 'checkout')
    with mock.patch.dict(
      os.environ,
      {'MUJOCO_PATH': str(sdk), 'MUJOCO_PLUGIN_PATH': str(plugins)},
      clear=True,
    ):
      with mock.patch.object(self.command, '_build_mujoco') as build:
        paths = self.command._find_mujoco()
    build.assert_not_called()
    self.assertEqual(paths, (str(lib), str(include), str(plugins), None))
    self.assertEqual(self.command._sdk_root, str(sdk))

  def test_prepared_sdist_skips_codegen_but_needs_external_sdk(self):
    _, source = self.fake_source(prepared=True)
    globals_ = self.command._prepare_source.__globals__
    with mock.patch.dict(
      globals_, {'__file__': str(source.parent / 'setup.py')}
    ):
      with mock.patch.object(self.module['subprocess'], 'check_call') as call:
        result = self.command._prepare_source()
    call.assert_not_called()
    self.assertEqual(result, str(source))
    self.assertIsNone(self.command._repo_root)
    with mock.patch.dict(os.environ, {}, clear=True):
      with mock.patch.object(self.command, '_build_mujoco') as build:
        with self.assertRaisesRegex(RuntimeError, 'prepared sdist requires'):
          self.command._find_mujoco()
      build.assert_not_called()

  def test_checkout_stages_codegen_without_writing_to_source(self):
    root, source = self.fake_source()
    globals_ = self.command._prepare_source.__globals__
    calls = []

    def generate(args, *, stdout, env):
      calls.append((args, env['PYTHONPATH']))
      stdout.write(Path(args[1]).name)

    with mock.patch.dict(
      globals_, {'__file__': str(source.parent / 'setup.py')}
    ):
      with mock.patch.object(
        self.module['subprocess'], 'check_call', side_effect=generate
      ):
        staged = Path(self.command._prepare_source())
    self.assertEqual(self.command._repo_root, str(root))
    self.assertEqual(staged, self.tmp / 'build' / 'source' / 'mujoco')
    for script, output in zip(
      (
        'generate_enum_traits.py',
        'generate_function_traits.py',
        'generate_spec_bindings.py',
      ),
      ('enum_traits.h', 'function_traits.h', 'specs.cc.inc'),
    ):
      self.assertFalse((source / output).exists())
      self.assertEqual((staged / output).read_text(), script)
    self.assertEqual(
      [Path(args[1]) for args, _ in calls],
      [
        staged / 'codegen' / name
        for name in (
          'generate_enum_traits.py',
          'generate_function_traits.py',
          'generate_spec_bindings.py',
        )
      ],
    )
    self.assertTrue(all(args[0] == sys.executable for args, _ in calls))
    self.assertTrue(
      all(path.split(os.pathsep)[0] == str(staged) for _, path in calls)
    )
    self.assertEqual(
      (staged / 'cmake/MujocoOptions.cmake').read_text(),
      'cmake/MujocoOptions.cmake',
    )
    self.assertEqual(
      (staged / 'simulate/CMakeLists.txt').read_text(),
      'simulate/CMakeLists.txt',
    )

  def test_native_build_installs_core_and_runtime_plugins(self):
    self.command._repo_root = str(self.tmp / 'checkout')
    for system, suffix, folder in (
      ('Darwin', '.dylib', 'lib'),
      ('Linux', '.so', 'lib'),
      ('Windows', '.dll', 'bin/Release'),
    ):
      with self.subTest(system=system):
        self.command._is_apple = system == 'Darwin'
        self.command.build_temp = str(self.tmp / system / 'build')
        lib = Path(self.command.build_temp) / 'native' / folder
        lib.mkdir(parents=True, exist_ok=True)
        expected = {
          f'{"" if system == "Windows" else "lib"}{plugin}{suffix}'
          for plugin in PLUGINS
        }
        for name in expected:
          (lib / name).touch()
        (lib / 'libelasticity.a').touch()
        with (
          mock.patch.dict(os.environ, {}, clear=True),
          mock.patch.object(
            self.module['platform'], 'system', return_value=system
          ),
          mock.patch.object(self.module['subprocess'], 'check_call') as call,
        ):
          self.command._build_mujoco()
        self.assertEqual(call.call_count, 4)
        build = call.call_args_list[1].args[0]
        self.assertEqual(
          build[build.index('--target') + 1 : build.index('--parallel')],
          ['mujoco', *PLUGINS],
        )
        self.assertEqual(
          [c.args[0][-1] for c in call.call_args_list[2:]],
          ['runtime', 'dev'],
        )
        sdk_plugins = Path(self.command._sdk_root) / 'mujoco_plugin'
        self.assertEqual(
          {path.name for path in sdk_plugins.iterdir()}, expected
        )

  def test_binding_cmake_uses_staged_source(self):
    staged = self.tmp / 'build/source/mujoco'
    self.command._binding_source = str(staged)
    self.command._mujoco_library_path = str(self.tmp / 'sdk/lib')
    self.command._mujoco_include_path = str(self.tmp / 'sdk/include')
    self.command._mujoco_framework_path = None
    with (
      mock.patch.object(self.module['subprocess'], 'check_call') as call,
      mock.patch('builtins.print'),
    ):
      self.command._configure_cmake()
    configure = call.call_args_list[0].args[0]
    self.assertEqual(configure[-1], str(staged))
    self.assertIn(f'-DCMAKE_MODULE_PATH:PATH={staged / "cmake"}', configure)

  def test_runtime_plugins_are_copied_into_wheel(self):
    for system, name in (
      ('Darwin', 'libelasticity.dylib'),
      ('Linux', 'libelasticity.so'),
      ('Windows', 'elasticity.dll'),
    ):
      with self.subTest(system=system):
        plugins = self.tmp / 'plugins' / system
        plugins.mkdir(parents=True)
        (plugins / name).touch()
        self.command._mujoco_plugins_path = str(plugins)
        self.command.build_lib = str(self.tmp / 'wheel' / system)
        with mock.patch.object(
          self.module['platform'], 'system', return_value=system
        ):
          self.command._copy_plugin_libraries()
        wheel = self.tmp / 'wheel' / system / 'mujoco/plugin'
        self.assertTrue((wheel / name).is_file())

  def test_missing_core_extensions_fail_but_optional_ones_skip(self):
    self.assertTrue(
      {f'mujoco.{name}' for name in CORE}
      <= {extension.name for extension in self.extensions}
    )
    for name in CORE:
      with self.subTest(name=name):
        with self.assertRaisesRegex(RuntimeError, name):
          self.command.build_extension(
            self.module['CMakeExtension'](f'mujoco.{name}')
          )
    with mock.patch('builtins.print') as warning:
      self.command.build_extension(
        self.module['CMakeExtension']('mujoco._render_filament')
      )
    self.assertIn('Skipping', warning.call_args.args[0])


if __name__ == '__main__':
  unittest.main()
