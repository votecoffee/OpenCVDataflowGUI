from glob import glob
from setuptools import find_packages, setup

setup(name="opencv_bin_picking_fixture", version="0.1.0", packages=find_packages(exclude=["test"]),
      data_files=[("share/ament_index/resource_index/packages", ["resource/opencv_bin_picking_fixture"]),
                  ("share/opencv_bin_picking_fixture", ["package.xml", "README.md", "LICENSE"]),
                  ("share/opencv_bin_picking_fixture/launch", glob("launch/*.launch.py"))],
      install_requires=["setuptools"], zip_safe=False, maintainer="votecoffee", maintainer_email="votecoffee@outlook.com",
      description="Additive simulation-only bin and RGB-D fixture with explicitly assisted grasping", license="MIT",
      entry_points={"console_scripts": ["fixture_prepare = opencv_bin_picking_fixture.prepare:main",
                                         "fixture_run = opencv_bin_picking_fixture.run:main",
                                         "fixture_bridge = opencv_bin_picking_fixture.node:main"]})
