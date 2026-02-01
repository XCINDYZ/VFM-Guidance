"""
Guided Variational Flow Matching Model

This module implements guided VFM with classifier guidance
for improved sample quality and controllability.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable


class GuidedVFM(nn.Module):
    """
    Guided Variational Flow Matching model.
    
    This model incorporates guidance from a classifier or energy function
    to improve generation quality or achieve specific objectives.
    
    Args:
        input_dim (int): Dimension of the input data
        hidden_dim (int): Dimension of hidden layers
        num_layers (int): Number of hidden layers
        time_embedding_dim (int): Dimension of time embedding
        guidance_scale (float): Scale of the guidance signal
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        num_layers: int = 3,
        time_embedding_dim: int = 32,
        guidance_scale: float = 1.0
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.time_embedding_dim = time_embedding_dim
        self.guidance_scale = guidance_scale
        
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
    
    def guided_forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        guidance_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute the guided vector field using a guidance function.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            t: Time tensor of shape (batch_size, 1) or (batch_size,)
            guidance_fn: Function that takes x and returns guidance gradient
        
        Returns:
            Guided vector field of shape (batch_size, input_dim)
        """
        # Base vector field
        v = self.forward(x, t)
        
        # Add guidance if x requires grad
        if x.requires_grad:
            guidance = guidance_fn(x)
            v = v + self.guidance_scale * guidance
        
        return v
    
    def sample(
        self,
        num_samples: int,
        num_steps: int = 100,
        guidance_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Generate samples with optional guidance.
        
        Args:
            num_samples: Number of samples to generate
            num_steps: Number of discretization steps
            guidance_fn: Optional guidance function
            device: Device to run on
        
        Returns:
            Generated samples of shape (num_samples, input_dim)
        """
        # Start from base distribution
        x = torch.randn(num_samples, self.input_dim, device=device)
        
        # Solve ODE using Euler method
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t = torch.ones(num_samples, 1, device=device) * (step * dt)
            
            if guidance_fn is not None:
                x.requires_grad_(True)
                v = self.guided_forward(x, t, guidance_fn)
                x = x.detach() + v.detach() * dt
            else:
                v = self.forward(x, t)
                x = x + v * dt
            
        return x
