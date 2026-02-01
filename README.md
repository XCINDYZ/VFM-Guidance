# VFM-Guidance

A comprehensive repository for Variational Flow Matching (VFM) research code. This repository contains implementations of various VFM models with detailed notebooks for reproduction and experimentation.

## 📋 Overview

This repository provides:
- Multiple implementations of Variational Flow Matching models
- Jupyter notebooks for different VFM variants
- Modular code structure for easy experimentation
- Ready-to-use training and inference scripts

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/XCINDYZ/VFM-Guidance.git
   cd VFM-Guidance
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   ```

## 📚 Repository Structure

```
VFM-Guidance/
├── notebooks/              # Jupyter notebooks for different VFM models
│   ├── basic_vfm/         # Basic variational flow matching
│   ├── conditional_vfm/   # Conditional VFM models
│   └── guided_vfm/        # Guided VFM models
├── src/                   # Source code
│   ├── models/            # Model implementations
│   └── utils/             # Utility functions
├── data/                  # Data directory
│   ├── raw/               # Raw data files
│   └── processed/         # Processed data files
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 📓 Notebooks

### Basic VFM
- **`notebooks/basic_vfm/`**: Introduction to basic variational flow matching concepts and implementations

### Conditional VFM
- **`notebooks/conditional_vfm/`**: Conditional flow matching models for controlled generation

### Guided VFM
- **`notebooks/guided_vfm/`**: Guided variational flow matching with different guidance strategies

## 🔧 Usage

### Running Notebooks

1. Start Jupyter:
   ```bash
   jupyter notebook
   ```

2. Navigate to the `notebooks/` directory and open any notebook

3. Run the cells sequentially to:
   - Load and prepare data
   - Initialize models
   - Train the VFM models
   - Visualize results

### Using the Code Directly

```python
# Example: Import VFM models
from src.models import BasicVFM

# Initialize model
model = BasicVFM(input_dim=2, hidden_dim=128)

# Your training code here...
```

## 📊 Data

Place your data files in:
- `data/raw/`: Original, unprocessed data
- `data/processed/`: Cleaned and preprocessed data

The data directories are excluded from git by default. Download or generate data as described in individual notebooks.

## 🛠️ Development

### Adding New Models

1. Create a new Python file in `src/models/`
2. Implement your model class
3. Add a corresponding notebook in the appropriate `notebooks/` subdirectory
4. Document the model and its usage

### Code Style

- Follow PEP 8 conventions
- Add docstrings to all functions and classes
- Include type hints where applicable

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{vfm-guidance,
  author = {Your Name},
  title = {VFM-Guidance: Variational Flow Matching Research Code},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/XCINDYZ/VFM-Guidance}
}
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the maintainers.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PyTorch team for the deep learning framework
- Research community for VFM developments