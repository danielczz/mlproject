from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    """This function return list of requirements.txt."""
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
        
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    
    return requirements
    
# Contains all Metadata related to my Project
setup(
name="mlproject",
version="0.0.1",
author="danielczz",
author_email="daniel.cespedes.zorob@gmail.com",
packages=find_packages(),
install_requires=get_requirements("requirements.txt")
)


# Python packages are available on "Python PyPi"
# Setup.py is the setup script for all activity in building, distributing, and installing modules
# Helps me manage my ML application as a Package
