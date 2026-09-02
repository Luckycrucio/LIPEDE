from glob import glob
import os
from setuptools import find_packages, setup

package_name = "lipede"


def tree_data(source):
    return [
        (os.path.join("share", package_name, os.path.dirname(path)), [path])
        for path in glob(source + "/**/*", recursive=True)
        if os.path.isfile(path)
    ]

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
        ("share/" + package_name + "/models", glob("models/*")),
    ] + tree_data("vendor"),
    install_requires=["setuptools"],
    zip_safe=False,
    maintainer="FAST-LIPEDE maintainer",
    maintainer_email="maintainer@example.com",
    description="Real-time LARS semantic filtering for Ouster PointCloud2 streams.",
    license="Apache-2.0",
    entry_points={"console_scripts": [
        "lipede_node = lipede.online_node:main",
        "lipede_offline = lipede.offline_node:main",
    ]},
)
