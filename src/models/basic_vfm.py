"""
Basic Variational Flow Matching Model

This module implements the basic VFM architecture for learning
continuous normalizing flows.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class BasicVFM(nn.Module):
    """
    Basic Variational Flow Matching model.
    
    This model learns a vector field that transforms a simple base distribution
    (e.g., Gaussian) to a complex target distribution.
    
    Args:
        input_dim (int): Dimension of the input data
        hidden_dim (int): Dimension of hidden layers
        num_layers (int): Number of hidden layers
        time_embedding_dim (int): Dimension of time embedding
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 3,
        time_embedding_dim: int = 32
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.time_embedding_dim = time_embedding_dim
        
        # Time embedding network
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )
        
        # Vector field network
        layers = []
        in_dim = input_dim + time_embedding_dim
        
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else input_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.SiLU())
            in_dim = out_dim
            
        self.net = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the vector field at position x and time t.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            t: Time tensor of shape (batch_size, 1) or (batch_size,)
        
        Returns:
            Vector field of shape (batch_size, input_dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
            
        # Embed time
        t_embed = self.time_embed(t)
        
        # Concatenate input and time embedding
        xt = torch.cat([x, t_embed], dim=-1)
        
        # Compute vector field
        v = self.net(xt)
        
        return v
    
    def sample(
        self,
        num_samples: int,
        num_steps: int = 100,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Generate samples by solving the ODE.
        
        Args:
            num_samples: Number of samples to generate
            num_steps: Number of discretization steps
            device: Device to run on
        
        Returns:
            Generated samples of shape (num_samples, input_dim)
        """
        # Start from base distribution (standard Gaussian)
        x = torch.randn(num_samples, self.input_dim, device=device)
        
        # Solve ODE using Euler method
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.ones(num_samples, 1, device=device) * (step * dt)
            v = self.forward(x, t)
            x = x + v * dt
            
        return x
