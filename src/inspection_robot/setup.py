from setuptools import find_packages, setup
from glob import glob

package_name = 'inspection_robot'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Phongsakon Sithong',
    maintainer_email='phongsakonsithong@gmail.com',
    description='Nodes for autonomous patrol navigation and real-time defect detection using YOLO',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'defect_detector_server = inspection_robot.defect_detector_server:main',
            'patrol_navigator = inspection_robot.patrol_navigator:main',
            'ble_localizer = inspection_robot.ble_localizer:main',
        ],
    },
)
