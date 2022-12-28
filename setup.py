from setuptools import setup
from pathlib import Path


with open('VERSION') as f:
    version = f.read().strip()

with open('requirements.txt') as f:
    install_requirements = f.read().splitlines()

setup(
    name="lightcone",
    version=version,
    packages=["lightcone"],
    python_requires='>=3.8.0',
    # dependencies
    install_requires=install_requirements,
    tests_require=[
        "pytest",
        "sphinx",
        "sphinx_rtd_theme",
    ],
    # metadata for upload to PyPI
    author="Tobias Windisch",
    author_email="tobias.windisch@posteo.de",
    description="A framework to explore the latent space of autoencoders implemented in torch",
    license="Apache-2.0",
    long_description=(Path(__file__).parent / "README.md").read_text(),
    long_description_content_type='text/markdown',
    keywords="graphical models",
    url="https://github.com/windisch/lightcone",
)
