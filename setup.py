from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('requirements_gui.txt') as f:
    requirements_gui = f.read().splitlines()

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="renalsegmentor",
    version="1.3.6",
    description="Segment kidneys from MRI data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alexdaniel654/Renal_Segmentor",
    license="GPL-3.0",

    python_requires='>=3.8, <4',
    packages=find_packages(),
    install_requires=requirements,
    extras_require={'gui': [requirements_gui]},
    include_package_data=True,
    classifiers=[
        'Development Status :: 5 - Production/Stable',

        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Environment :: Console',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',

        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',

        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
    ],
)
