from distutils.core import setup

setup(
    name='FormalAbstraction',
    version='0.1',
    packages=['Models', 'ApprxSimulation', 'Demo_file', 'label_abstraction'],
    license='BSD-3',
    author='shaesaert',
    author_email='haesaert@caltech.edu',
    description='Routines for formal abstraction and controller synthesis',
    package_data={'label_abstraction': ['binaries/mac/scheck2','binaries/mac/ltl2ba']}
)
