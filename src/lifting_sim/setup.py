from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'lifting_sim'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.sdf')),
        (os.path.join('share', package_name, 'models/m5fly'), glob('models/m5fly/*')),
        (os.path.join('share', package_name, 'models/pingpong_ball'), glob('models/pingpong_ball/*')),
        (os.path.join('share', package_name, 'models/ground_camera'), glob('models/ground_camera/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='uburyo',
    maintainer_email='uburyo@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'controller = lifting_sim.controller:main',
        ],
    },
)
