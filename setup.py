#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

test_requirements = ['pytest>=3', ]

setup(
    author="Thomas Vaitses Fontanari",
    author_email='tvfontanari@gmail.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Gaussian Mixture Model over histograms",
    entry_points={
        'console_scripts': [
            'histgmm=histgmm.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme,
    include_package_data=True,
    keywords='histgmm',
    name='histgmm',
    packages=find_packages(include=['histgmm', 'histgmm.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/thomasvf/histgmm',
    version='0.1.0',
    zip_safe=False,
)
