#!/usr/bin/env python
import os

from setuptools import setup, find_packages


if __name__ == "__main__":

    base_dir = os.path.dirname(__file__)
    src_dir = os.path.join(base_dir, "src")

    setup(

        package_dir={'': 'src'},
        packages=find_packages(where='src'),
        include_package_data=True,
        zip_safe=False,
    )
