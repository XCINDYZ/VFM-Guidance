"""
Training utilities for VFM models.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from tqdm import tqdm


def compute_loss(
    model: nn.Module,
    x1: torch.Tensor,
    sigma: float = 0.01
) -> torch.Tensor:
    """
    Compute the flow matching loss.
    
    Args:
        model: VFM model
        x1: Target samples of shape (batch_size, input_dim)
        sigma: Noise level for conditional flow matching
    
    Returns:
        Loss value
    """
    batch_size = x1.shape[0]
    device = x1.device
    
    # Sample source points from base distribution
    x0 = torch.randn_like(x1)
    
    # Sample time uniformly
    t = torch.rand(batch_size, 1, device=device)
    
    # Interpolate between x0 and x1
    x_t = (1 - t) * x0 + t * x1 + sigma * torch.randn_like(x1)
    
    # Compute target velocity (conditional flow)
    v_target = x1 - x0
    
    # Compute predicted velocity
    v_pred = model(x_t, t)
    
    # MSE loss
    loss = ((v_pred - v_target) ** 2).mean()
    
    return loss


def train_vfm(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    device: str = 'cpu',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train a VFM model.
    
    Args:
        model: VFM model to train
        dataloader: DataLoader with training data
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        verbose: Whether to print progress
    
    Returns:
        Dictionary with training history
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'losses': []
    }
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        iterator = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}') if verbose else dataloader
        
        for batch in iterator:
            x1 = batch[0].to(device)
            
            optimizer.zero_grad()
            loss = compute_loss(model, x1)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if verbose and isinstance(iterator, tqdm):
                iterator.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / num_batches
        history['losses'].append(avg_loss)
        
        if verbose:
            print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}')
    
    return history


def train_conditional_vfm(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    device: str = 'cpu',
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Train a conditional VFM model.
    
    Args:
        model: Conditional VFM model to train
        dataloader: DataLoader with training data and conditions
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        verbose: Whether to print progress
    
    Returns:
        Dictionary with training history
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    history = {
        'losses': []
    }
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        iterator = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}') if verbose else dataloader
        
        for batch in iterator:
            x1, condition = batch[0].to(device), batch[1].to(device)
            
            batch_size = x1.shape[0]
            x0 = torch.randn_like(x1)
            t = torch.rand(batch_size, 1, device=device)
            
            x_t = (1 - t) * x0 + t * x1
            v_target = x1 - x0
            
            optimizer.zero_grad()
            v_pred = model(x_t, t, condition)
            loss = ((v_pred - v_target) ** 2).mean()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if verbose and isinstance(iterator, tqdm):
                iterator.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / num_batches
        history['losses'].append(avg_loss)
        
        if verbose:
            print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}')
    
    return history
