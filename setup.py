from setuptools import setup, find_packages

setup(
    name="azul_zero",
    version="0.1.0",
    author="Alberto Pozo",
    description="Deep Reinforcement Learning agent for the Azul board game using MCTS + PyTorch",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch",
        "numpy",
        "gym",
        "tensorboard",
    ],
    python_requires=">=3.11",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)