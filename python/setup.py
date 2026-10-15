# Copyright 2022 DeepMind Technologies Limited
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
"""Install script for MuJoCo."""

import fnmatch
import logging
import os
import platform
import random
import re
import shutil
import string
import subprocess
import sys
import sysconfig

import setuptools
from setuptools.command import build_ext
from setuptools.command import install_scripts

MUJOCO_CMAKE = 'MUJOCO_CMAKE'
MUJOCO_CMAKE_ARGS = 'MUJOCO_CMAKE_ARGS'
MUJOCO_PATH = 'MUJOCO_PATH'
MUJOCO_PLUGIN_PATH = 'MUJOCO_PLUGIN_PATH'

EXT_PREFIX = 'mujoco.'
REQUIRED_EXTENSIONS = frozenset(
  {
    'mujoco._callbacks',
    'mujoco._constants',
    'mujoco._enums',
    'mujoco._errors',
    'mujoco._functions',
    'mujoco._render',
    'mujoco._rollout',
    'mujoco._simulate',
    'mujoco._specs',
    'mujoco._structs',
  }
)


def get_long_description():
  """Creates a long description for the package from bundled markdown files."""
  current_dir = os.path.dirname('__file__')
  with open(os.path.join(current_dir, 'README.md'), encoding='utf-8') as f:
    description = f.read()
  try:
    with open(os.path.join(current_dir, 'LICENSES_THIRD_PARTY.md'), encoding='utf-8') as f:
      description = f'{description}\n{f.read()}'
  except FileNotFoundError:
    pass
  return description


def get_mujoco_lib_pattern():
  if platform.system() == 'Windows':
    return 'mujoco.lib'
  elif platform.system() == 'Darwin':
    return 'libmujoco.*.dylib'
  else:
    return 'libmujoco.so.*'


def get_external_lib_patterns():
  if platform.system() == 'Windows':
    return ['mujoco.dll']
  elif platform.system() == 'Darwin':
    return ['libmujoco.*.dylib']
  else:
    return ['libmujoco.so.*']


def get_plugin_lib_patterns():
  if platform.system() == 'Windows':
    return ['*.dll']
  elif platform.system() == 'Darwin':
    return ['lib*.dylib']
  else:
    return ['lib*']


def start_and_end(iterable):
  it = iter(iterable)
  while True:
    try:
      first = next(it)
      second = next(it)
      yield first, second
    except StopIteration:
      return


def tokenize_quoted_substr(input_string, quote_char, placeholders=None):
  """Replace quoted substrings with random text placeholders with no spaces."""
  # Matches quote characters not proceded with a backslash.
  pattern = re.compile(r'(?<!\\)' + quote_char)
  quote_positions = [m.start() for m in pattern.finditer(input_string)]
  if len(quote_positions) % 2:
    raise ValueError(f'unbalanced quotes {quote_char}...{quote_char}')

  output_string = ''
  placeholders = placeholders if placeholders is not None else dict()
  prev_end = -1
  for start, end in start_and_end(quote_positions):
    output_string += input_string[prev_end + 1 : start]
    while True:
      placeholder = ''.join(random.choices(string.ascii_lowercase, k=5))
      if placeholder not in input_string and placeholder not in output_string:
        break
    output_string += placeholder
    placeholders[placeholder] = input_string[start + 1 : end]
    prev_end = end
  output_string += input_string[prev_end + 1 :]

  return output_string, placeholders


def parse_cmake_args_from_environ(env_var_name=MUJOCO_CMAKE_ARGS):
  """Parses CMake arguments from an environment variable."""
  raw_args = os.environ.get(env_var_name, '').strip()
  unquoted, placeholders = tokenize_quoted_substr(raw_args, '"')
  unquoted, placeholders = tokenize_quoted_substr(unquoted, "'", placeholders)
  parts = re.split(r'\s+', unquoted.strip())
  out = []
  for part in parts:
    for k, v in placeholders.items():
      part = part.replace(k, v)
    part = part.replace('\\"', '"').replace("\\'", "'")
    if part:
      out.append(part)
  return out


def get_macos_arch_args():
  if platform.system() != 'Darwin' or not os.environ.get('ARCHFLAGS'):
    return []
  archs = [
    arch
    for arch in ('x86_64', 'arm64')
    if f'-arch {arch}' in os.environ['ARCHFLAGS']
  ]
  return [f'-DCMAKE_OSX_ARCHITECTURES={";".join(archs)}'] if archs else []


class CMakeExtension(setuptools.Extension):
  """A Python extension that has been prebuilt by CMake.

  We do not want distutils to handle the build process for our extensions, so
  so we pass an empty list to the super constructor.
  """

  def __init__(self, name):
    super().__init__(name, sources=[])


class BuildCMakeExtension(build_ext.build_ext):
  """Uses CMake to build extensions."""

  def run(self):
    self._is_apple = platform.system() == 'Darwin'
    self._binding_source = self._prepare_source()
    (
        self._mujoco_library_path,
        self._mujoco_include_path,
        self._mujoco_plugins_path,
        self._mujoco_framework_path,
    ) = self._find_mujoco()
    self._configure_cmake()
    for ext in self.extensions:
      assert ext.name.startswith(EXT_PREFIX)
      self.build_extension(ext)
    self._copy_external_libraries()
    self._copy_mujoco_headers()
    self._copy_plugin_libraries()
    self._copy_studio_assets()
    if self._is_apple:
      self._copy_mjpython()

  def _prepare_source(self):
    source = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mujoco')
    root = os.path.dirname(os.path.dirname(source))
    root_files = (
      'CMakeLists.txt',
      'include/mujoco/mujoco.h',
      'cmake/MujocoOptions.cmake',
      'simulate/CMakeLists.txt',
      'plugin/sdf/CMakeLists.txt',
    )
    self._repo_root = (
      root
      if all(os.path.isfile(os.path.join(root, name)) for name in root_files)
      else None
    )
    prepared_files = (
      'enum_traits.h',
      'function_traits.h',
      'specs.cc.inc',
      'cmake/MujocoOptions.cmake',
      'simulate/CMakeLists.txt',
    )
    if all(
      os.path.isfile(os.path.join(source, name)) for name in prepared_files
    ):
      return source
    if not self._repo_root:
      raise RuntimeError(
        'Python binding source is incomplete: use a full MuJoCo checkout '
        'or a prepared release sdist with MUJOCO_PATH and MUJOCO_PLUGIN_PATH'
      )

    staged = os.path.join(os.path.abspath(self.build_temp), 'source', 'mujoco')
    shutil.copytree(source, staged, dirs_exist_ok=True)
    shutil.copytree(
      os.path.join(root, 'cmake'),
      os.path.join(staged, 'cmake'),
      dirs_exist_ok=True,
    )
    shutil.copytree(
      os.path.join(root, 'simulate'),
      os.path.join(staged, 'simulate'),
      dirs_exist_ok=True,
    )
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join(
      filter(None, (staged, env.get('PYTHONPATH')))
    )
    for script, output in (
      ('generate_enum_traits.py', 'enum_traits.h'),
      ('generate_function_traits.py', 'function_traits.h'),
      ('generate_spec_bindings.py', 'specs.cc.inc'),
    ):
      with open(
        os.path.join(staged, output), 'w', encoding='utf-8'
      ) as generated:
        subprocess.check_call(
          [sys.executable, os.path.join(staged, 'codegen', script)],
          stdout=generated,
          env=env,
        )
    return staged

  def _build_parallel_args(self):
    if os.environ.get('CMAKE_BUILD_PARALLEL_LEVEL'):
      return []
    return ['--parallel', str(min(os.cpu_count() or 1, 4))]

  def _build_mujoco(self):
    cmake = os.environ.get(MUJOCO_CMAKE, 'cmake')
    build_cfg = 'Debug' if self.debug else 'Release'
    native_build = os.path.join(os.path.abspath(self.build_temp), 'native')
    self._sdk_root = os.path.join(os.path.abspath(self.build_temp), 'sdk')
    cmake_args = [
      '-S',
      self._repo_root,
      '-B',
      native_build,
      f'-DCMAKE_BUILD_TYPE:STRING={build_cfg}',
      f'-DCMAKE_INSTALL_PREFIX:PATH={self._sdk_root}',
      '-DCMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL=OFF',
      '-DBUILD_TESTING:BOOL=OFF',
      '-DMUJOCO_BUILD_EXAMPLES:BOOL=OFF',
      '-DMUJOCO_BUILD_SIMULATE:BOOL=OFF',
      '-DMUJOCO_BUILD_TESTS:BOOL=OFF',
      '-DMUJOCO_TEST_PYTHON_UTIL:BOOL=OFF',
      '-DMUJOCO_BUILD_STUDIO:BOOL=OFF',
      '-DMUJOCO_USE_FILAMENT:BOOL=OFF',
      '-DMUJOCO_WITH_USD:BOOL=OFF',
      '-DMUJOCO_BUILD_MACOS_FRAMEWORKS:BOOL=OFF',
    ]
    if platform.system() != 'Windows':
      origin = '@loader_path' if self._is_apple else '$ORIGIN'
      cmake_args.extend(
        [
          '-DCMAKE_BUILD_WITH_INSTALL_RPATH:BOOL=ON',
          f'-DCMAKE_INSTALL_RPATH:STRING={origin};{origin}/..',
        ]
      )
    cmake_args.extend(get_macos_arch_args())
    cmake_args.extend(parse_cmake_args_from_environ())
    if platform.system() == 'Windows':
      cmake_args = [arg.replace('\\', '/') for arg in cmake_args]
    subprocess.check_call([cmake, *cmake_args])
    subprocess.check_call(
      [
        cmake,
        '--build',
        native_build,
        '--config',
        build_cfg,
        '--target',
        'mujoco',
        'actuator',
        'elasticity',
        'sensor',
        'sdf_plugin',
        *self._build_parallel_args(),
      ]
    )
    for component in ('runtime', 'dev'):
      subprocess.check_call(
        [
          cmake,
          '--install',
          native_build,
          '--prefix',
          self._sdk_root,
          '--config',
          build_cfg,
          '--component',
          component,
        ]
      )

    system = platform.system()
    lib_dir = os.path.join(
      native_build, 'bin' if system == 'Windows' else 'lib'
    )
    if os.path.isdir(os.path.join(lib_dir, build_cfg)):
      lib_dir = os.path.join(lib_dir, build_cfg)
    suffix = (
      '.dll' if system == 'Windows' else '.dylib' if self._is_apple else '.so'
    )
    plugin_dir = os.path.join(self._sdk_root, 'mujoco_plugin')
    os.makedirs(plugin_dir, exist_ok=True)
    for plugin in ('actuator', 'elasticity', 'sensor', 'sdf_plugin'):
      filename = ('' if system == 'Windows' else 'lib') + plugin + suffix
      src = os.path.join(lib_dir, filename)
      if not os.path.isfile(src):
        raise RuntimeError(f'Native MuJoCo plugin not built: {src}')
      shutil.copyfile(src, os.path.join(plugin_dir, filename))

  def _find_mujoco(self):
    if MUJOCO_PATH in os.environ or MUJOCO_PLUGIN_PATH in os.environ:
      if not (
        os.environ.get(MUJOCO_PATH) and os.environ.get(MUJOCO_PLUGIN_PATH)
      ):
        raise RuntimeError(
          'Set both MUJOCO_PATH and MUJOCO_PLUGIN_PATH to nonempty SDK paths'
        )
      self._sdk_root = os.environ[MUJOCO_PATH]
      plugin_path = os.environ[MUJOCO_PLUGIN_PATH]
    else:
      if not self._repo_root:
        raise RuntimeError(
          'A prepared sdist requires MUJOCO_PATH and MUJOCO_PLUGIN_PATH; '
          'build from a full MuJoCo checkout for an automatic engine build'
        )
      self._build_mujoco()
      plugin_path = os.path.join(self._sdk_root, 'mujoco_plugin')
    library_path = None
    include_path = None
    for directory, subdirs, filenames in os.walk(self._sdk_root):
      if self._is_apple and 'mujoco.framework' in subdirs:
        return (
          os.path.join(directory, 'mujoco.framework/Versions/A'),
          os.path.join(directory, 'mujoco.framework/Headers'),
          plugin_path,
          directory,
        )
      if fnmatch.filter(filenames, get_mujoco_lib_pattern()):
        library_path = directory
      if os.path.exists(os.path.join(directory, 'mujoco/mujoco.h')):
        include_path = directory
      if library_path and include_path:
        return library_path, include_path, plugin_path, None
    raise RuntimeError('Cannot find MuJoCo library and/or include paths')

  def _copy_external_libraries(self):
    dst = os.path.dirname(self.get_ext_fullpath(self.extensions[0].name))
    for directory, _, filenames in os.walk(self._sdk_root):
      for pattern in get_external_lib_patterns():
        for filename in fnmatch.filter(filenames, pattern):
          shutil.copyfile(
              os.path.join(directory, filename), os.path.join(dst, filename)
          )

  def _copy_plugin_libraries(self):
    dst = os.path.join(
        os.path.dirname(self.get_ext_fullpath(self.extensions[0].name)),
        'plugin',
    )
    os.makedirs(dst, exist_ok=True)
    for directory, _, filenames in os.walk(self._mujoco_plugins_path):
      for pattern in get_plugin_lib_patterns():
        for filename in fnmatch.filter(filenames, pattern):
          shutil.copyfile(
              os.path.join(directory, filename), os.path.join(dst, filename)
          )

  def _copy_mujoco_headers(self):
    dst = os.path.join(
        os.path.dirname(self.get_ext_fullpath(self.extensions[0].name)),
        'include/mujoco',
    )
    os.makedirs(dst, exist_ok=True)
    for directory, _, filenames in os.walk(self._mujoco_include_path):
      rel_dir = os.path.relpath(directory, self._mujoco_include_path)

      # Skip third-party directories
      if rel_dir.startswith(('SDL2', 'math', 'misc')):
        continue

      for filename in fnmatch.filter(filenames, '*.h'):
        rel_file_path = os.path.relpath(
            os.path.join(directory, filename), self._mujoco_include_path
        )

        # Skip third-party files in the root include path (for framework case)
        if rel_dir == '.' and not (
            filename == 'mujoco.h' or filename.startswith('mj')
        ):
          continue

        # Reconstruct destination path preserving structure
        # Strip leading 'mujoco/' if present to avoid duplicate 'mujoco/mujoco/'
        if rel_file_path.startswith('mujoco/'):
          target_rel_path = rel_file_path[len('mujoco/') :]
        else:
          target_rel_path = rel_file_path

        target_dst = os.path.join(dst, target_rel_path)
        os.makedirs(os.path.dirname(target_dst), exist_ok=True)
        shutil.copyfile(os.path.join(directory, filename), target_dst)


  def _copy_studio_assets(self):
    assets_src = None
    for directory, subdirs, _ in os.walk(self._sdk_root):
      if 'assets' in subdirs:
        candidate = os.path.join(directory, 'assets')
        if os.path.exists(os.path.join(candidate, 'fontawesome-webfont.ttf')):
          assets_src = candidate
          break

    if assets_src:
      dst = os.path.join(
          os.path.dirname(self.get_ext_fullpath(self.extensions[0].name)),
          'experimental/studio/assets',
      )
      if os.path.exists(dst):
        shutil.rmtree(dst)
      shutil.copytree(assets_src, dst)
    else:
      print("Warning: Studio assets not found in MuJoCo SDK. Skipping.")

  def _copy_mjpython(self):
    src_dir = os.path.join(os.path.dirname(__file__), 'mujoco/mjpython')
    dst_contents_dir = os.path.join(
        os.path.dirname(self.get_ext_fullpath(self.extensions[0].name)),
        'MuJoCo_(mjpython).app/Contents',
    )
    os.makedirs(dst_contents_dir, exist_ok=True)
    shutil.copyfile(
        os.path.join(src_dir, 'Info.plist'),
        os.path.join(dst_contents_dir, 'Info.plist'),
    )

    dst_bin_dir = os.path.join(dst_contents_dir, 'MacOS')
    os.makedirs(dst_bin_dir, exist_ok=True)
    mjpython = os.path.join(self.build_temp, 'mjpython')
    if not os.path.isfile(mjpython):
      mjpython = os.path.join(
        self.build_temp, 'Debug' if self.debug else 'Release', 'mjpython'
      )
    shutil.copyfile(mjpython, os.path.join(dst_bin_dir, 'mjpython'))
    os.chmod(os.path.join(dst_bin_dir, 'mjpython'), 0o755)

    dst_resources_dir = os.path.join(dst_contents_dir, 'Resources')
    os.makedirs(dst_resources_dir, exist_ok=True)
    shutil.copyfile(
        os.path.join(src_dir, 'mjpython.icns'),
        os.path.join(dst_resources_dir, 'mjpython.icns'),
    )

  def _configure_cmake(self):
    """Check for CMake."""
    cmake = os.environ.get(MUJOCO_CMAKE, 'cmake')
    build_cfg = 'Debug' if self.debug else 'Release'
    cmake_module_path = os.path.join(self._binding_source, 'cmake')
    cmake_args = [
        f'-DPython3_ROOT_DIR:PATH={sys.prefix}',
        f'-DPython3_EXECUTABLE:STRING={sys.executable}',
        f'-DCMAKE_MODULE_PATH:PATH={cmake_module_path}',
        f'-DCMAKE_BUILD_TYPE:STRING={build_cfg}',
        f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY:PATH={os.path.abspath(self.build_temp)}',
        (
            f'-DCMAKE_INTERPROCEDURAL_OPTIMIZATION:BOOL={"OFF" if self.debug else "ON"}'
        ),
        '-DCMAKE_Fortran_COMPILER:STRING=',
        '-DBUILD_TESTING:BOOL=OFF',
    ]

    if self._mujoco_framework_path is not None:
      cmake_args.extend([
          f'-DMUJOCO_FRAMEWORK_DIR:PATH={self._mujoco_framework_path}',
      ])
    else:
      cmake_args.extend([
          f'-DMUJOCO_LIBRARY_DIR:PATH={self._mujoco_library_path}',
          f'-DMUJOCO_INCLUDE_DIR:PATH={self._mujoco_include_path}',
      ])

    if platform.system() != 'Windows':
      cmake_args.extend([
          f'-DPython3_LIBRARY={sysconfig.get_paths()["stdlib"]}',
          f'-DPython3_INCLUDE_DIR={sysconfig.get_paths()["include"]}',
      ])
    cmake_args.extend(get_macos_arch_args())

    cmake_args.extend(parse_cmake_args_from_environ())
    os.makedirs(self.build_temp, exist_ok=True)

    if platform.system() == 'Windows':
      cmake_args = [arg.replace('\\', '/') for arg in cmake_args]

    print('Configuring CMake with the following arguments:')
    for arg in cmake_args:
      print(f'    {arg}')
    subprocess.check_call(
        [cmake]
        + cmake_args
        + [self._binding_source],
        cwd=self.build_temp,
    )

    print('Building all extensions with CMake')
    subprocess.check_call(
        [cmake, '--build', '.', *self._build_parallel_args(), '--config', build_cfg],
        cwd=self.build_temp,
    )

  def build_extension(self, ext):
    dest_path = self.get_ext_fullpath(ext.name)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # Reconstruct relative path from extension name to support nested extensions
    rel_ext_name = ext.name[len(EXT_PREFIX):]
    rel_path = rel_ext_name.replace('.', '/')
    filename = os.path.basename(dest_path)
    rel_dir = os.path.dirname(rel_path)

    build_path = os.path.join(self.build_temp, rel_dir, filename)
    if not os.path.isfile(build_path):
      build_path = os.path.join(
        self.build_temp,
        rel_dir,
        'Debug' if self.debug else 'Release',
        filename,
      )
    if os.path.isfile(build_path):
      shutil.copyfile(build_path, dest_path)
    elif ext.name in REQUIRED_EXTENSIONS:
      raise RuntimeError(f'Required MuJoCo extension not built: {ext.name}')
    else:
      print(f'Warning: Extension {ext.name} was not built by CMake. Skipping.')


class InstallScripts(install_scripts.install_scripts):
  """Strips file extension from executable scripts whose names end in `.py`."""

  def run(self):
    super().run()
    oldfiles = self.outfiles
    files = set(oldfiles)
    self.outfiles = []
    for oldfile in oldfiles:
      if oldfile.endswith('.py'):
        newfile = oldfile[:-3]
      else:
        newfile = oldfile

      renamed = False
      if newfile not in files and not os.path.exists(newfile):
        if not self.dry_run:
          os.rename(oldfile, newfile)
        renamed = True

      if renamed:
        logging.info(
            'Renaming %s script to %s',
            os.path.basename(oldfile),
            os.path.basename(newfile),
        )
        self.outfiles.append(newfile)
        files.remove(oldfile)
        files.add(newfile)
      else:
        self.outfiles.append(oldfile)


setuptools.setup(
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    cmdclass=dict(
        build_ext=BuildCMakeExtension,
        install_scripts=InstallScripts,
    ),
    ext_modules=[
        CMakeExtension('mujoco._callbacks'),
        CMakeExtension('mujoco._constants'),
        CMakeExtension('mujoco._enums'),
        CMakeExtension('mujoco._errors'),
        CMakeExtension('mujoco._functions'),
        CMakeExtension('mujoco._render'),
        CMakeExtension('mujoco._render_filament'),
        CMakeExtension('mujoco._rollout'),
        CMakeExtension('mujoco._simulate'),
        CMakeExtension('mujoco._specs'),
        CMakeExtension('mujoco._structs'),
        # Studio extensions
        CMakeExtension('mujoco.experimental.studio.native_viewer_cc'),
        CMakeExtension('mujoco.experimental.studio.ux'),
        CMakeExtension('mujoco.experimental.studio.sim'),
        CMakeExtension('mujoco.experimental.studio.window'),
        # Studio web extensions
        CMakeExtension('mujoco.experimental.studio.web.headless_ui'),
        CMakeExtension('mujoco.experimental.studio.web.state_payload'),
        # ImGui/ImPlot extensions
        CMakeExtension('mujoco.experimental.dear_imgui.dear_imgui'),
        CMakeExtension('mujoco.experimental.implot.implot'),
    ],
    scripts=['mujoco/mjpython/mjpython.py']
    if platform.system() == 'Darwin'
    else [],
)
