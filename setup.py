from numpy import get_include as np_get_include
from distutils.core import setup
from Cython.Build import cythonize

_VERSION = "0.2.2"

with open("README", "r") as readme:
    long_desc = readme.read()


setup(
    version=_VERSION,
    name="whittle-levinson-durbin-recursion",
    ext_modules=cythonize("levinson/clevinson.pyx"),
    packages=["levinson", "test"],
    include_dirs=np_get_include(),
    author="Ryan J. Kinnear",
    author_email="Ryan@Kinnear.ca",
    description=("Implementations of the Levinson-Durbin algorithm and "
                 "Whittle's multivariate (block-toeplitz) version for "
                 "estimating VAR(p) models"),
    long_description=long_desc,
    url="https://github.com/RJTK/Levinson-Durbin-Recursion",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering"],
    license="LICENSE"
)
