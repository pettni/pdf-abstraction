from setuptools import setup, find_packages

setup(
    name='BelST',
    version='0.1',
    packages=find_packages(),    license='BSD-3',
    author='shaesaert',
    author_email='s.haesaert@tue.nl',
    description='Routines for LTL planning in uncertain environments',
    package_data={'BeLST': ['binaries/mac/scheck2','binaries/mac/ltl2ba']}
)
