# Medical AI Analysis Tools

A collection of advanced medical AI tools for image analysis and intelligent medical document processing.

## Projects

### 1. ResNet50 BALF Analysis
Deep learning framework for analyzing bronchoalveolar lavage fluid (BALF) cell populations using ResNet50. Provides automated differential cell counting and clinical interpretability features.

[View ResNet50 BALF Analysis Documentation](./Resnet50-image-analysis/README.md)

Key Features:
- Automated cell population analysis
- Advanced training optimizations
- Clinical interpretability tools
- Grad-CAM visualizations

### 2. MediTron RAG
Retrieval-Augmented Generation (RAG) system specialized for medical document processing and analysis.

[View MediTron RAG Documentation](./meditron-rag/README.md)

Key Features:
- Document processing and analysis
- Medical knowledge retrieval
- Intelligent query processing
- Integration with medical imaging analysis

## Repository Structure
```
.
├── Resnet50-image-analysis/    # BALF cell analysis project
│   ├── src/                    # Source code for image analysis
│   └── data/                   # Example datasets
│
├── meditron-rag/               # Medical document RAG system
│   ├── src/                    # Source code for RAG
│   │   ├── utils/             # Utility functions
│   │   ├── model/             # Model implementations
│   │   └── data/              # Vector store and embeddings
│   │       └── vector_store/  # Chroma DB storage
```

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/ggkasman/BALF.git
cd BALF
```

2. Set up individual projects:
   - Follow the setup instructions in each project's README
   - Install project-specific dependencies as needed
