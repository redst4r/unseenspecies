from setuptools import setup

setup(name='unseenspecies',
      version=0.1,
      description='Python functions related to unseen species estimation',
      url='http://github.com/redst4r/unseenspecies/',
      author='redst4r',
      maintainer='redst4r',
      maintainer_email='redst4r@web.de',
      license='BSD 2-Clause License',
      packages=['unseenspecies'],
      install_requires=[
          'numpy',
          'scipy',
          'tqdm',
          'rpy2',
          'pandas'
          # 'toolz',
          ],
      zip_safe=False)
