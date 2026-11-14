from setuptools import find_packages, setup

setup(
    name="opencv_dataflow_planning",
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/opencv_dataflow_planning"]),
        ("share/opencv_dataflow_planning", ["package.xml", "README.md", "LICENSE"]),
    ],
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="votecoffee",
    maintainer_email="votecoffee@outlook.com",
    description="Plan-only MoveIt request/review and simulation benchmark adapters.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "plan_bridge = opencv_dataflow_planning.bridge:main",
            "planning_benchmark = opencv_dataflow_planning.benchmark:main",
        ]
    },
)
