"""
Data utilities for generating and processing datasets.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional


def generate_toy_data(
    name: str = 'two_moons',
    num_samples: int = 1000,
    noise: float = 0.1
) -> torch.Tensor:
    """
    Generate toy 2D datasets for testing VFM models.
    
    Args:
        name: Dataset name ('two_moons', 'swiss_roll', 'circles', 'gaussian')
        num_samples: Number of samples to generate
        noise: Noise level
    
    Returns:
        Generated samples of shape (num_samples, 2)
    """
    if name == 'two_moons':
        from sklearn.datasets import make_moons
        data, _ = make_moons(n_samples=num_samples, noise=noise)
        
    elif name == 'swiss_roll':
        from sklearn.datasets import make_swiss_roll
        data, _ = make_swiss_roll(n_samples=num_samples, noise=noise)
        data = data[:, [0, 2]]  # Use only x and z dimensions
        
    elif name == 'circles':
        from sklearn.datasets import make_circles
        data, _ = make_circles(n_samples=num_samples, noise=noise, factor=0.5)
        
    elif name == 'gaussian':
        data = np.random.randn(num_samples, 2) * 0.5
        
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
    # Normalize to roughly [-1, 1] range
    data = (data - data.mean(axis=0)) / (data.std(axis=0) + 1e-8)
    
    return torch.from_numpy(data).float()


def plot_samples(
    samples: torch.Tensor,
    title: str = 'Generated Samples',
    figsize: Tuple[int, int] = (8, 8),
    save_path: Optional[str] = None
) -> None:
    """
    Plot 2D samples.
    
    Args:
        samples: Tensor of shape (num_samples, 2)
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save the figure
    """
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    
    plt.figure(figsize=figsize)
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=10)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    plt.close()  # Free memory resources


def create_dataloader(
    data: torch.Tensor,
    batch_size: int = 128,
    shuffle: bool = True
) -> torch.utils.data.DataLoader:
    """
    Create a PyTorch DataLoader from data tensor.
    
    Args:
        data: Input data tensor
        batch_size: Batch size
        shuffle: Whether to shuffle the data
    
    Returns:
        DataLoader object
    """
    dataset = torch.utils.data.TensorDataset(data)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )
