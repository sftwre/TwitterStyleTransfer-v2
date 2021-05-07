from setuptools import find_packages
from setuptools import setup


with open('./requirements.txt') as f:
    REQUIRED_PACKAGES = f.read().splitlines()


setup(
    name='seq2seq',
    version='1.0',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Training | TwitterStyleTransfer'
)
