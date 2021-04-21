from setuptools import find_packages
from setuptools import setup
from pip.req import parse_requirements

REQUIRED_PACKAGES = [
    'google-cloud-storage>=1.14.0',
    'pandas>=0.23.4',
    'tensorboard==2.4.0',
    'torchtext==0.8.0',
    'nltk==3.5'
]

setup(
    name='trainer',
    version='1.0',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Training | TwitterStyleTransfer'
)
