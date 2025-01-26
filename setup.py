from setuptools import setup

setup(
    name="AltTextGenerator",
    python_requires=">=3.12",
    install_requires=[
        "Pillow>=11.1.0,<12.0.0",
        "transformers>=4.48.0,<5.0.0",
        "torch>=2.5.1,<3.0.0",
        "pytest~=7.4.4",
        "setuptools~=75.1.0",
    ],
)