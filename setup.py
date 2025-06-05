from importlib.metadata import entry_points
from setuptools import setup, find_packages



def readme():
  with open('README.md') as f:
    return f.read()

setup(
  name    = "datautilities",  
  use_scm_version=True,
  setup_requires=['setuptools_scm', 'setuptools>=61'],
  description = "A Python package containing various data utility modules.",
  long_description=readme(),
  author  = "Johan Hofmans",
  author_email="johan.hofmans@xxx.com",
  license="LGPL",
  packages = find_packages(),
  install_requires=[
    'pulp',
    'scipy',
    'pandas',
    'rdkit',
    'matplotlib',
    'chembl_structure_pipeline',
    'parameterized',
    'pytest',
    'boto3',
    'requests'
  ],
  package_data={
    'datautilities.chemalerts':['*.tsv']
  },
  extras_require={
    'snowflake': [
        'snowflake-connector-python[pandas]'
    ],
    'test': [
        'pytest>=6.2.2',
        'parameterized>=0.8.1'
    ]
  },
  zip_safe=False,
)
