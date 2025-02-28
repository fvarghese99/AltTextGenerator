# setup.py

from setuptools import setup, find_packages
import platform
import subprocess
import torch

# Function to detect OS and GPU type
def detect_gpu_or_acceleration():
    try:
        # Check for NVIDIA GPU
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return "NVIDIA"
    except FileNotFoundError:
        pass

    try:
        # Check for AMD GPU
        result = subprocess.run(["rocm-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return "AMD"
    except FileNotFoundError:
        pass

    if platform.system() == "Darwin":
        # Check for MPS (Metal Performance Shaders) support on macOS
        if torch.backends.mps.is_available():
            return "MPS"

    return "NONE"


# Get the acceleration type (GPU or MPS)
acceleration_type = detect_gpu_or_acceleration()

# Specify dependencies based on OS and acceleration type
if platform.system() == "Windows" and acceleration_type == "NVIDIA":
    torch_dependencies = [
        "torch==2.5.1+cu126",
        "torchvision==0.20.1+cu126",
        "torchaudio==2.5.1+cu126",
    ]
elif platform.system() == "Windows" and acceleration_type == "AMD":
    torch_dependencies = [
        "torch==2.5.1+rocm5.5",
        "torchvision==0.20.1+rocm5.5",
        "torchaudio==2.5.1+rocm5.5",
    ]
elif platform.system() == "Darwin" and acceleration_type == "MPS":
    torch_dependencies = [
        "torch>=2.5.1,<3.0.0",  # Standard PyTorch works with MPS
        "torchvision>=0.20.0,<0.21.0",
        "torchaudio>=2.5.0,<3.0.0",
    ]
else:
    # Default CPU dependencies
    torch_dependencies = [
        "torch>=2.5.1,<3.0.0",
        "torchvision>=0.20.0,<0.21.0",
        "torchaudio>=2.5.0,<3.0.0",
    ]

# General dependencies
general_dependencies = [
    "Pillow>=11.1.0,<12.0.0",
    "transformers>=4.48.0,<5.0.0",
]

# Combine dependencies
dependencies = general_dependencies + torch_dependencies

setup(
    name="AltTextGenerator",
    version="1.0.0",
    description="A Python tool to generate alt text for images using the BLIP2 model.",
    author="fvarghese99",
    url="https://github.com/fvarghese99/AltTextGenerator",
    packages=find_packages(),
    python_requires=">=3.12",  # Require Python 3.12 or above
    install_requires=dependencies,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "alttextgenerator=src.main:main",
        ],
    },
)