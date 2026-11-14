from setuptools import find_packages, setup

name = "opencv_dataflow_pick_backend"
setup(
    name=name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + name]),
        ("share/" + name, ["package.xml", "README.md", "LICENSE", "native_acceptance.json"]),
    ],
    install_requires=["setuptools", "numpy"],
    zip_safe=True,
    maintainer="votecoffee",
    maintainer_email="votecoffee@outlook.com",
    description="Simulation-only ROS planning/execution primitives; app-owned pick workflow",
    license="MIT",
    entry_points={"console_scripts": ["sim_pick_backend = " + name + ".worker:main"]},
)
