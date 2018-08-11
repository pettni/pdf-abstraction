from setuptools import setup, find_packages

setup(
    name='FormalAbstraction',
    version='0.2',
    packages=find_packages(),    license='BSD-3',
    author='shaesaert',
    author_email='haesaert@caltech.edu',
    description='Routines for formal abstraction and controller synthesis',
    package_data={'best': ['binaries/mac/scheck2','binaries/mac/ltl2ba']}
)
