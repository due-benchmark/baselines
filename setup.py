# -*- coding: utf-8 -*-

"""
Set up benchmarker package.

To use the 'upload' functionality of this file, you must do:
    pip install twine
"""

import io
import os
import sys
import warnings
from pathlib import Path
from shutil import rmtree
from typing import Union

from setuptools import Command, find_packages, setup


def parse_requirements(req_path: Path):
    """Parse requirements

    Args:
        req_path: path to requirements file

    Returns:
        list of requirement strings in form of "library" or "library==0.0.0"
    """

    # Parse requirements from file - no need to specify them many times
    # Any dependencies defined in includes are omitted.
    # Only dependencies in provided file are considered
    # This is intended behavior
    parsed_requirements = []
    with open(req_path) as f:
        for line in f:
            content = line.split("#")[0].strip()  # remove comments
            if "://" in content:  # URL
                warnings.warn(f"Ignoring '{content}' requirement. Setuptools does not support URLs.")
            elif len(content) > 0 and content[0] != '-':  # non empty and not a command
                parsed_requirements.append(content)

    return parsed_requirements


def read_requirements(file_path: Union[str, Path] = 'requirements/core.txt') -> list:
    """Get required packages for this module to be executed.

    Args:
        file_path: the requirements file path.

    Returns:
        list of required packages

    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if not file_path.exists():
        return []
    req_path = file_path.resolve()
    return parse_requirements(req_path)


# Package meta-data.
NAME = 'benchmarker'
DESCRIPTION = 'DUE-BASELINES models for 2D document processing'
URL = ''
AUTHOR = 'Applica.ai'
REQUIRES_PYTHON = '>=3.6.0'
VERSION = None
REQUIRES = read_requirements() + read_requirements('requirements/pre.txt')

# Create dict for extras on the basis of requirements/extras/[name].txt pattern
EXTRAS = {}
for file_path in Path('requirements/extras/').glob("*.txt"):
    name = file_path.stem
    reqs = read_requirements(file_path)
    if len(reqs) > 0:
        EXTRAS[name] = reqs
    else:
        warnings.warn(f'Extras group {name} does not have any valid requirements and will not be available.')

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    with io.open(Path('README.md').resolve(), encoding='utf-8') as f:
        long_description = '\n' + f.read()
except FileNotFoundError:
    long_description = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}

if not VERSION:
    with open(Path(NAME + '/__version__.py').resolve()) as f:
        exec(f.read(), about)
    VERSION = about['__version__']
else:
    about['__version__'] = VERSION


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Print things in bold.

        Args:
            s: string to print in bold.

        """
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(Path('dist').resolve())
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system(
            '{0} setup.py sdist bdist_wheel --universal'.format(
                sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(about['__version__']))
        os.system('git push --tags')

        sys.exit()


if __name__ == '__main__':
    # Where the magic happens:
    setup(
        name=NAME,
        version=about['__version__'],
        description=DESCRIPTION,
        long_description=long_description,
        long_description_content_type='text/markdown',
        author=AUTHOR,
        # author_email=EMAIL,
        python_requires=REQUIRES_PYTHON,
        url=URL,
        packages=['benchmarker'] + ['benchmarker.' + pkg for pkg in find_packages('benchmarker')],
        install_requires=REQUIRES,
        extras_require=EXTRAS,
        include_package_data=True,
        license='PROPRIETARY Applica',
        classifiers=[
            # Trove classifiers
            'Development Status :: 3 - Alpha',
            'License :: PROPRIETARY Applica',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: Implementation :: PyPy'
        ],
        # $ setup.py publish support.
        cmdclass={
            'upload': UploadCommand,
        },
    )
