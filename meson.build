project(
  'motep',
  'c',
  version: '0.1.0',
)

add_project_arguments(
  '-O3',
  '-ffast-math',
  '-march=native',
  language: 'c',
)

py = import('python').find_installation(pure: false)

install_subdir(
  'motep',
  install_dir: py.get_install_dir(),
  exclude_directories: ['__pycache__'],
)

inc_numpy = run_command(
  py,
  ['-c', 'import numpy; print(numpy.get_include())'],
  check: true
).stdout().strip()

py.extension_module(
  '_mtp_cext',
  [
    'motep/potentials/mtp/cext/mtp_cext.c',
    'motep/potentials/mtp/cext/mtp_cext_module.c',
  ],
  include_directories: include_directories(
    inc_numpy,
    'motep/potentials/mtp/cext',
  ),
  subdir: 'motep/potentials/mtp/cext',
  install: true,
)
