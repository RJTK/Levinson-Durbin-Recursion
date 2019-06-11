from numpy import get_include as np_get_include
from distutils.core import setup
from Cython.Build import cythonize

setup(
    name="Levinson-Recursion",
    language_level="Py3",
    ext_modules=cythonize("clevinson.pyx"),
    include_dirs=np_get_include()
)
