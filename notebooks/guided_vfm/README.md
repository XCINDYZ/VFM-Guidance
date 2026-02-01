# Guided VFM Notebooks

This directory contains notebooks demonstrating Guided Variational Flow Matching.

## Notebooks

- **guided_vfm_demo.ipynb**: Guided VFM with different guidance strategies

## What is Guided VFM?

Guided VFM incorporates guidance signals (e.g., from a classifier, energy function, or other objectives) to steer the generation process. This improves sample quality or achieves specific generation objectives.

## Key Concepts

- **Classifier Guidance**: Using classifier gradients to guide generation
- **Energy-Based Guidance**: Using energy functions to guide the flow
- **Guidance Scale**: Controlling the strength of guidance
- **Adaptive Guidance**: Adjusting guidance during generation

## Use Cases

- Improving sample quality
- Targeting specific regions of the distribution
- Multi-objective generation
- Constrained generation

## Running the Notebooks

1. Ensure you have installed all dependencies (see main README)
2. Start Jupyter Notebook from the repository root
3. Navigate to this directory and open the notebook
4. Run cells sequentially
