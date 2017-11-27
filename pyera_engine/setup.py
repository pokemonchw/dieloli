import os
import sys

python_dir = os.path.dirname(sys.executable).replace('\\', '/')

from cx_Freeze import setup, Executable

os.environ['TCL_LIBRARY'] = python_dir + '/tcl/tcl8.6'
os.environ['TK_LIBRARY'] = python_dir + '/tcl/tk8.6'

# Dependencies are automatically detected, but it might need
# fine tuning.
buildOptions = dict(
    optimize=2,
    packages=[],
    excludes=[],
    include_files=[python_dir + '/DLLs/tcl86t.dll', python_dir + '/DLLs/tk86t.dll'],
)

import sys

base = None

executables = [
    Executable('pyera.py', base=base)
]

setup(name='pyera',
      version='0.1',
      description='',
      options=dict(build_exe=buildOptions),
      executables=executables)
