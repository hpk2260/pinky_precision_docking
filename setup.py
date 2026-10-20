from glob import glob
import os
from setuptools import find_packages, setup

package_name = 'pinky_precision_docking'

setup(
    name=package_name,
    version='5.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@todo.todo',
    description='Precision ArUco auto-docking package for Pinky Pro mobile robot',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'precision_docking_node = pinky_precision_docking.precision_docking_node:main',
            'generate_marker = pinky_precision_docking.generate_aruco_marker:main',
            'run_docking_scenarios = pinky_precision_docking.run_docking_scenarios:main',
            'run_corridor_scenarios = pinky_precision_docking.run_corridor_scenarios:main',
            'run_arena_mission = pinky_precision_docking.run_arena_mission:main',
            'sound_effects = pinky_precision_docking.sound_effects:main',
        ],
    },
)
