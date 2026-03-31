import subprocess

# List of packages with specific versions to avoid conflicts
packages = [
    "tensorflow>=2.12.0,<2.22.0",
    "flwr>=1.4.0,<2.0.0",
    "protobuf>=3.20.3,<5.0.0",
    "numpy>=1.23.5,<3.0.0",
    "matplotlib>=3.7.1,<4.0.0",
    "kagglehub>=0.1.0,<1.0.0",
    "seaborn>=0.12.2,<1.0.0",
    "scikit-learn>=1.3.0,<2.0.0"
]

# Upgrade pip to the latest version
subprocess.run(["pip", "install", "--upgrade", "pip"], check=True)

# Install each package
for package in packages:
    try:
        subprocess.run(["pip", "install", package], check=True)
        print(f"Successfully installed {package}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package}: {e}")