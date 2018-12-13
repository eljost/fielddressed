#!/usr/bin/env python3

from setuptools import setup, find_packages
import sys

if sys.version_info.major < 3:
    raise SystemExit("Python 3 is required!")

setup(
    name="fielddressed",
    version="0.0.1",
    description="Plotting of field dressed states.",
    url="https://github.com/eljost/fielddressed",
    maintainer="Johannes Steinmetzer",
    maintainer_email="johannes.steinmetzer@uni-jena.de",
    license="GPL 3",
    platforms=["unix"],
    packages=find_packages(),
    package_data={"fielddressed": ["assets/*.css", ],},
    install_requires=[
        "dash",
        "dash-html-components",
        "dash-core-components",
        "dash-table",
        "plotly",
        "pyyaml",
        "numpy",
    ],
    entry_points={
        "console_scripts": [
            "fielddressed = fielddressed.main:run",
        ]
    },
)
