# Quick Start Guide

This guide will help you get started with the VFM-Guidance repository quickly.

## Installation

### Option 1: Using the Setup Script (Recommended for Linux/Mac)

```bash
# Clone the repository
git clone https://github.com/XCINDYZ/VFM-Guidance.git
cd VFM-Guidance

# Run the setup script
bash setup.sh
```

### Option 2: Manual Installation

```bash
# Clone the repository
git clone https://github.com/XCINDYZ/VFM-Guidance.git
cd VFM-Guidance

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Test

Verify your installation works:

```bash
python -c "from src.models import BasicVFM; import torch; model = BasicVFM(); print('Installation successful!')"
```

## Running Your First Notebook

1. Activate your virtual environment (if not already activated):
   ```bash
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

3. In your browser, navigate to: `notebooks/basic_vfm/basic_vfm_demo.ipynb`

4. Run all cells (Cell → Run All)

## Next Steps

After running the basic VFM notebook, try:

1. **Conditional VFM**: `notebooks/conditional_vfm/conditional_vfm_demo.ipynb`
   - Learn about conditional generation
   - Control sample generation with class labels

2. **Guided VFM**: `notebooks/guided_vfm/guided_vfm_demo.ipynb`
   - Implement guidance mechanisms
   - Improve generation quality

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`, make sure:
- Your virtual environment is activated
- All dependencies are installed: `pip install -r requirements.txt`

### CUDA Issues

If you want to use GPU but get CUDA errors:
- Check PyTorch installation: `python -c "import torch; print(torch.cuda.is_available())"`
- Reinstall PyTorch with CUDA support: See https://pytorch.org/get-started/locally/

### Jupyter Not Starting

If Jupyter doesn't start:
- Install explicitly: `pip install jupyter notebook`
- Try using JupyterLab instead: `pip install jupyterlab` then `jupyter lab`

## Getting Help

- Check the main [README.md](README.md) for detailed documentation
- Look at individual notebook README files in each subdirectory
- Open an issue on GitHub if you encounter problems

## Using the Models in Your Code

```python
import torch
from src.models import BasicVFM
from src.utils import generate_toy_data, train_vfm, create_dataloader

# Generate data
data = generate_toy_data('two_moons', num_samples=1000)

# Create model
model = BasicVFM(input_dim=2, hidden_dim=128)

# Train
dataloader = create_dataloader(data, batch_size=128)
history = train_vfm(model, dataloader, num_epochs=50, device='cpu')

# Generate samples
model.eval()
samples = model.sample(num_samples=500, num_steps=100, device='cpu')

print(f"Generated {len(samples)} samples")
```

Happy coding! 🚀
