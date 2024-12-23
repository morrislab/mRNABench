from setuptools import setup, find_packages

# Parse README
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="mrna_bench",
    version="0.0.1",
    author="IanShi1996",
    description="Benchmarking suite for mRNA.",
    long_description_content_type="text/markdown",
    long_description=long_description,
    url="https://github.com/IanShi1996/mRNABench",
    install_requires=[
        "torch>=2.0.0",
        "scikit-learn",
    ],
    python_requires='>=3.10',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    packages=find_packages(
        where=".",
        exclude=[]
    ),
)
