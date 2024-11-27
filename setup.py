#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [ ]

test_requirements = [ ]

setup(
    author="Jimin Tan",
    author_email='tanjimin@nyu.edu',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Self-supervised analysis for multiplexed immunofluorescence imaging",
    entry_points={
        'console_scripts': [
            'canvas=canvas.cli:main',
        ],
    },
    install_requires=requirements,
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='canvas',
    name='canvas',
    packages=find_packages(include=['canvas', 'canvas.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/tanjimin/canvas',
    version='0.0.1',
    zip_safe=False,
)
