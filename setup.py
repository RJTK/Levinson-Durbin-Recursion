from numpy import get_include as np_get_include
from distutils.core import setup
from Cython.Build import cythonize

setup(
    name="whittle-levinson-durbin-recursion-RJTK",
    language_level="Py3",
    author="Ryan J. Kinnear",
    author_email="Ryan@Kinnear.ca",
    description=("Implementations of the Levinson-Durbin algorithm and "
                 "Whittle's multivariate (block-toeplitz) version for "
                 "estimating VAR(p) models"),
    url="https://github.com/RJTK/Levinson-Durbin-Recursion",
    version="0.1.0",
    ext_modules=cythonize("levinson/clevinson.pyx"),
    include_dirs=np_get_include(),
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
                 "Operating System :: OS Independent",
                 "Topic :: Scientific/Engineering"]

)
