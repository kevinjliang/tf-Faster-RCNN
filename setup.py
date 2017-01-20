from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(
      name = 'tf-Faster-RCNN',
      version='0.1',
      description='TensorFlow implementation of Faster R-CNN',
      author='Kevin Liang',
      ext_modules=cythonize("Lib/*.pyx"),
      include_dirs=[numpy.get_include()]
)