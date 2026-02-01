# Contributing to VFM-Guidance

Thank you for your interest in contributing to VFM-Guidance! This document provides guidelines for researchers who want to add their own VFM implementations or improve existing code.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a new branch for your feature
4. Make your changes
5. Test your changes
6. Submit a pull request

## Adding a New VFM Model

### 1. Create the Model Class

Add your model to `src/models/`:

```python
# src/models/your_vfm.py
import torch
import torch.nn as nn

class YourVFM(nn.Module):
    """
    Brief description of your VFM variant.
    
    Args:
        input_dim (int): Dimension of input data
        # Add other parameters
    """
    
    def __init__(self, input_dim: int = 2, ...):
        super().__init__()
        # Your initialization code
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the vector field.
        
        Args:
            x: Input tensor
            t: Time tensor
            
        Returns:
            Vector field
        """
        # Your forward pass
        
    def sample(self, num_samples: int, **kwargs) -> torch.Tensor:
        """
        Generate samples.
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            Generated samples
        """
        # Your sampling code
```

### 2. Update Model __init__.py

Add your model to `src/models/__init__.py`:

```python
from .your_vfm import YourVFM

__all__ = [..., 'YourVFM']
```

### 3. Create a Demonstration Notebook

Create a notebook in `notebooks/your_model_category/`:

- Follow the structure of existing notebooks
- Include clear explanations
- Provide visualizations
- Compare with other methods if relevant

### 4. Add Documentation

Create a README in your notebook directory explaining:
- What your model does
- Key concepts
- When to use it
- How it differs from other models

## Code Style Guidelines

### Python Code

- Follow PEP 8 style guidelines
- Use type hints for function arguments and return values
- Add docstrings to all classes and functions
- Keep functions focused and modular
- Use meaningful variable names

Example:

```python
def train_vfm(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Train a VFM model.
    
    Args:
        model: VFM model to train
        dataloader: DataLoader with training data
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        device: Device to run training on
        
    Returns:
        Dictionary containing training history
    """
    # Implementation
```

### Notebooks

- Start with a clear title and overview
- Organize into logical sections with markdown headers
- Add explanatory text between code cells
- Include visualizations where appropriate
- End with conclusions and suggestions for experimentation

## Testing Your Changes

Before submitting a pull request:

1. **Test imports**: Ensure all imports work correctly
   ```python
   from src.models import YourVFM
   model = YourVFM()
   ```

2. **Test forward pass**: Verify model can process data
   ```python
   import torch
   x = torch.randn(10, 2)
   t = torch.rand(10, 1)
   output = model(x, t)
   ```

3. **Test sampling**: Verify sample generation works
   ```python
   samples = model.sample(num_samples=100)
   ```

4. **Run your notebook**: Execute all cells to ensure they work

## Pull Request Process

1. **Update documentation**: Ensure README files are updated
2. **Add examples**: Include usage examples in docstrings
3. **Describe changes**: Write a clear PR description explaining:
   - What you added/changed
   - Why it's useful
   - How to use it
4. **Reference issues**: Link to relevant issues if applicable

## Adding Utilities

If you're adding utility functions:

1. Add to appropriate file in `src/utils/`
2. Update `src/utils/__init__.py`
3. Add docstrings with examples
4. Consider adding a notebook example

## Research Paper Implementations

If implementing a method from a paper:

1. Add citation in docstring:
   ```python
   """
   Implementation of [Method Name] from:
   
   Author et al., "Paper Title", Conference/Journal Year
   Link: https://arxiv.org/abs/...
   """
   ```

2. Note any differences from the paper
3. Include paper-specific hyperparameters as defaults
4. Add comparison results if available

## Questions?

- Open an issue for questions
- Tag it as "question" or "help wanted"
- Provide context about what you're trying to do

## Code of Conduct

- Be respectful and constructive
- Welcome newcomers
- Focus on the science and the code
- Give credit where it's due

Thank you for contributing to VFM-Guidance! 🎉
