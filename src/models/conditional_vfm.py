"""
Conditional Variational Flow Matching Model

This module implements conditional VFM for controlled generation
based on class labels or other conditioning information.
"""

import torch
import torch.nn as nn
from typing import Optional


class ConditionalVFM(nn.Module):
    """
    Conditional Variational Flow Matching model.
    
    This model extends basic VFM to support conditional generation,
    allowing control over the generated samples through conditioning.
    
    Args:
        input_dim (int): Dimension of the input data
        condition_dim (int): Dimension of the conditioning vector
        hidden_dim (int): Dimension of hidden layers
        num_layers (int): Number of hidden layers
        time_embedding_dim (int): Dimension of time embedding
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        condition_dim: int = 10,
        hidden_dim: int = 128,
        num_layers: int = 3,
        time_embedding_dim: int = 32
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.time_embedding_dim = time_embedding_dim
        
        # Time embedding network
        self.time_embed = nn.Sequential(
            nn.Linear(1, time_embedding_dim),
            nn.SiLU(),
            nn.Linear(time_embedding_dim, time_embedding_dim)
        )
        
        # Condition embedding network
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Vector field network
        layers = []
        in_dim = input_dim + time_embedding_dim + hidden_dim
        
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else input_dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.SiLU())
            in_dim = out_dim
            
        self.net = nn.Sequential(*layers)
        
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the conditional vector field.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            t: Time tensor of shape (batch_size, 1) or (batch_size,)
            condition: Conditioning tensor of shape (batch_size, condition_dim)
        
        Returns:
            Vector field of shape (batch_size, input_dim)
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
            
        # Embed time and condition
        t_embed = self.time_embed(t)
        c_embed = self.condition_embed(condition)
        
        # Concatenate input, time, and condition embeddings
        xtc = torch.cat([x, t_embed, c_embed], dim=-1)
        
        # Compute vector field
        v = self.net(xtc)
        
        return v
    
    def sample(
        self,
        condition: torch.Tensor,
        num_steps: int = 100,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Generate conditional samples.
        
        Args:
            condition: Conditioning tensor of shape (num_samples, condition_dim)
            num_steps: Number of discretization steps
            device: Device to run on
        
        Returns:
            Generated samples of shape (num_samples, input_dim)
        """
        num_samples = condition.shape[0]
        
        # Start from base distribution
        x = torch.randn(num_samples, self.input_dim, device=device)
        condition = condition.to(device)
        
        # Solve ODE using Euler method
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.ones(num_samples, 1, device=device) * (step * dt)
            v = self.forward(x, t, condition)
            x = x + v * dt
            
        return x
