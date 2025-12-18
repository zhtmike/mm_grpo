import os

from setuptools import find_packages, setup

# Read the version from gerl/version/version
version_folder = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(version_folder, "gerl/version/version"), "r") as f:
    __version__ = f.read().strip()

# Read the requirements from requirements.txt
with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

# Read the long description from README.md
with open("README.md", "r") as f:
    long_description = f.read()


PADDLEOCR_REQUIRES = ["paddlepaddle", "paddleocr>=3.0", "python-Levenshtein"]
VLLM_REQUIRES = ["vllm>=0.11.1"]

extras_require = {
    "paddleocr": PADDLEOCR_REQUIRES,
    "vllm": VLLM_REQUIRES,
}

setup(
    name="gerl",
    version=__version__,
    author="Leibniz CSI Research Lab",
    author_email="jzhoubc@connect.ust.hk, kwcheungad@connect.ust.hk, ychengw@connect.ust.hk",
    description="A library to support RL training for multi-modal generative models.",
    license="Apache 2.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/leibniz-csi/mm_grpo",
    packages=find_packages(
        exclude=("tests", "examples", "assets", "recipe", "scripts", "__pycache__")
    ),
    package_data={"gerl": ["version/version"]},
    install_requires=requirements,
    extras_require=extras_require,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
