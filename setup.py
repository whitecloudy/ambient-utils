from pathlib import Path
import setuptools

root = Path(__file__).parent.resolve()

with open(root / "README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Define core dependencies (minimal set)
core_requirements = [
    "numpy",
    "scipy",
    "torch",
    "Pillow",
]

# Define optional dependencies for full functionality
extras_require = {
    "all": [
        "wandb",
        "seaborn", 
        "torchvision",
        "diffusers",
        "s3fs",
        "webdataset",
        "datasets",
        "imagecorruptions",
    ],
    "dev": [
        "pytest",
        "black",
        "flake8",
    ],
    "docs": [
        "sphinx",
        "sphinx-rtd-theme",
    ],
}

setuptools.setup(
    name="ambient_utils",
    version="1.0.2",
    author="giannisdaras",
    author_email="daras.giannhs@gmail.com",
    description="Utility functions for learning generative models from corrupted data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/giannisdaras/ambient_utils",
    packages=setuptools.find_packages(),
    install_requires=core_requirements,
    extras_require=extras_require,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Intended Audience :: Science/Research",
    ],
    license="GPL-3.0",
    python_requires='>=3.6',
)