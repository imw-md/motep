project(
  'motep',
  'c',
  version: '0.1.0',
)

py = import('python').find_installation(pure: false)

install_subdir(
  'motep',
  install_dir: py.get_install_dir(),
  exclude_directories: ['__pycache__'],
)
