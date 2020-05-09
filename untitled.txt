from setuptools import setup
from setuptools import find_packages

setup(name='SkipGNN',
      version='0.1',
      description='SkipGNN: Predicting Molecular Interactions with Skip-Graph Networks',
      author='Kexin Huang',
      author_email='kexinhuang@hsph.harvard.edu',
      url='kexinhuang.com',
      download_url='https://github.com/kexinhuang12345/SkipGNN',
      license='BSD-3',
      install_requires=['numpy',
                        'torch',
                        'scipy'
                        ],
      package_data={'SkipGNN': ['README.md']},
      packages=find_packages())