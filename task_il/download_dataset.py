"""
Script to download CIFAR-100 dataset before running evolution.
Run this ONCE before the first evolution run.
"""

import os
import torch
from torchvision import datasets, transforms

# Get paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'data')
binary_data_dir = os.path.join(script_dir, 'dat', 'binary_cifar_inc5')

print("="*60)
print("CIFAR-100 Dataset Download Script")
print("="*60)
print(f"\nData directory: {data_dir}")
print(f"Binary data directory: {binary_data_dir}\n")

# Create directories
os.makedirs(data_dir, exist_ok=True)
os.makedirs(binary_data_dir, exist_ok=True)

# Download CIFAR-100
print("Downloading CIFAR-100 dataset...")
print("This may take a few minutes (~160MB)...\n")

mean = [0.5071, 0.4867, 0.4408]
std = [0.2675, 0.2565, 0.2761]

try:
    # Download training set
    print("Downloading training set...")
    train_data = datasets.CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    )
    print("✓ Training set downloaded!")

    # Download test set
    print("Downloading test set...")
    test_data = datasets.CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    )
    print("✓ Test set downloaded!")

    print("\n" + "="*60)
    print("✓ Dataset download completed!")
    print("="*60)
    print(f"\nDataset location: {data_dir}/cifar-100-python/")
    print("\nYou can now run: python evolve.py")
    print("="*60)

except Exception as e:
    print(f"\n❌ Error downloading dataset: {e}")
    print("\nPlease check:")
    print("1. Internet connection")
    print("2. Available disk space")
    print("3. Write permissions in:", data_dir)
