from setuptools import setup
from torch.utils import cpp_extension

setup(name='correlation',
      ext_modules=[cpp_extension.CppExtension('correlation',\
                                              ['correlation.cpp'],include_dirs=['./'])],\
      license='Apache License v2.0',
      cmdclass={'build_ext':cpp_extension.BuildExtension})