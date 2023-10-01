import os.path as op

from setuptools import setup, find_packages


# get the version (don't import mne here, so dependencies are not needed)
version = None
with open(op.join('p_kit', '_version.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')

with open('README.md', 'r', encoding="utf8") as fid:
    long_description = fid.read()

setup(name='p-kit',
      version=version,
      description='Probabilistic circuit simulator',
      url='https://github.com/IBM/p-kit',
      author='Gregoire Cattan, Anton Andreev',
      author_email='gregoire.cattan@ibm.com',
      license='BSD (3-clause)',
      packages=find_packages(),
      long_description=long_description,
      long_description_content_type='text/markdown',
      project_urls={
          'Documentation': 'https://github.com/IBM/p-kit/wiki',
          'Source': 'https://github.com/IBM/p-kit',
          'Tracker': 'https://github.com/IBM/p-kitissues/',
      },
      platforms='any',
      python_requires=">=3.9",
      install_requires=[
                        'numpy<1.27',
                        'cython==3.0.2',
                        'cvxpy==1.3.2',
                        'scipy==1.11.3',
                        'docplex>=2.21.207',
                        'matplotlib==3.8.0'
                        ],
      extras_require={
                      'tests': ['pytest', 'seaborn', 'flake8'],
                      },
      zip_safe=False,
)
