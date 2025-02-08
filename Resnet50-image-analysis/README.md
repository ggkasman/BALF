# BALF Leukocyte Proportion Analysis

Deep learning framework for regression-based analysis of bronchoalveolar lavage fluid (BALF) cell populations using ResNet50. 

## Project Structure
```
BALF/
├── src/                     # Source code
├── data/                    # Example data (training, validation, test)
```

## Features

- ResNet50-based BALF cell analysis
- Automated differential cell counting:
  - Neutrophils, Eosinophils, Lymphocytes, Macrophages
- Training enhancements:
  - One-cycle learning rate scheduling
  - LR range finder
  - Extensive data augmentation
- Clinical interpretability:
  - Grad-CAM visualization
  - Prediction/label comparison files

## Data Requirements

Expected directory structure for BALF analysis:
```
Data/
├── Training/
│   ├── Case_01/ 
│   │   ├── XXX_MOSAIC_..._0.png
│   │   └── labels.txt
├── Validation/
└── Testing/
```

Note: The repository includes a small set of example data.

## Setup
1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`

## Training Optimization

Implements advanced learning rate strategies for better convergence:
- **LR Range Test**: Finds optimal learning rate bounds (`lr_finder.py`)
- **1cycle Policy**: Uses cosine-annealed rate/momentum scheduling (`lr_one_cycle.py`)
- Configuration: Default max LR = 1e-5, div_factor=25

## Training Process

- 10 iterations of cross-validation
- Early stopping (patience=15)
- Two-phase training:
  1. Initial frozen backbone
  2. Full network fine-tuning
- Hyperparameter tracking via TensorBoard

## Usage

1. Learning Rate Analysis:
```bash
python src/lr_finder.py
```

2. Training with 1cycle:
```bash
python src/model_train.py
```

3. Generate Explanations:
```bash
python src/model_visualization.py
```

## Explainability

Provides clinical interpretability through:
- Grad-CAM visualizations of predictions
- Class activation maps for all 4 cell types
- Visual explanation saving alongside predictions

## References

This implementation builds upon the methodology from:

> van Huizen LMG, Blokker M, Rip Y, et al. (2023)  
> **Leukocyte differentiation in bronchoalveolar lavage fluids using higher harmonic generation microscopy and deep learning.**  
> PLoS ONE 18(6): e0279525.  
> https://doi.org/10.1371/journal.pone.0279525  
> [Full Text Available Here](https://pmc.ncbi.nlm.nih.gov/articles/PMC10298778/pdf/pone.0279525.pdf)

**Implementation Note**: This codebase provides a PyTorch implementation of the deep learning analysis described in the paper, adapted for clinical BALF cell differentiation using ResNet50 architecture rather than custom CNNs.

## License

Apache 2.0 - See [LICENSE.txt](LICENSE.txt) 
