from glob import glob

from setuptools import setup

setup(
    name="opencv_hand_eye_demo",
    version="0.1.0",
    packages=["opencv_hand_eye_demo"],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/opencv_hand_eye_demo"]),
        ("share/opencv_hand_eye_demo", ["package.xml", "README.md", "LICENSE"]),
        ("share/opencv_hand_eye_demo/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="votecoffee",
    maintainer_email="votecoffee@outlook.com",
    description="Isolated rendered ROS 2 eye-in-hand calibration example",
    license="MIT",
    entry_points={
        "console_scripts": [
            "hand_eye_prepare = opencv_hand_eye_demo.prepare:main",
            "hand_eye_run = opencv_hand_eye_demo.run:main",
            "hand_eye_capture = opencv_hand_eye_demo.capture:main",
        ]
    },
)
