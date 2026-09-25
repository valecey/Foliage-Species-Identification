# Species Detection Pipeline

A production-grade machine learning pipeline for detecting tree species in geospatial imagery using SVM classifiers and DINOv2 embeddings.

## Project Structure

```
species-detection-pipeline/
├── config/
│   ├── __init__.py
│   └── config.py              # Central configuration
├── data/
│   ├── __init__.py
│   ├── annotation.py          # Data annotation utilities
│   ├── loader.py              # Dataset loading utilities
│   └── preprocessor.py        # Data preprocessing
├── models/
│   ├── __init__.py
│   ├── dimensionality_reduction.py  # Dimensionality reduction methods
│   ├── embedding.py           # DINOv2 embedding generation
│   ├── predictor.py          # Batch prediction
│   └── svm_trainer.py        # SVM training with bias correction
├── utils/
│   ├── __init__.py
│   ├── geojson_converter.py   # GeoJSON format conversion
│   ├── label_studio.py       # Label Studio format conversion
│   ├── morphology.py         # Morphological operations
│   └── visualization.py      # Plotting utilities
├── scripts/
│   ├── __init__.py
│   ├── create_data.py        # Data creation utilities
│   ├── evaluate.py           # Evaluation entry point
│   └── train.py              # Training entry point
├── requirements.txt
└── README.md
```

## Features

- **Modular Architecture**: Separation of concerns with dedicated modules for data, models, and utilities
- **Production-Ready**: Proper error handling, logging, and configuration management
- **Exact Logic Preservation**: All original functionality maintained with identical implementation
- **Type Hints**: Enhanced code readability and IDE support
- **Configurable**: Centralized configuration for easy parameter tuning
- **Extensible**: Easy to add new species, models, or processing steps

## Installation

### Option 1: Install dependencies

```bash
git clone <repository-url>
cd species-detection-pipeline
pip install -r requirements.txt
```

## Usage

### Training Models

```bash
# Run training script
python scripts/train.py

# Optional: Specify config file
python scripts/train.py --config path/to/config.yaml
```

The training script will:
1. Load training datasets and embeddings
2. Train biased SVM models for each species
3. Optimize hyperparameters (gamma, C, class weights)
4. Save trained models to `lp3_first_run_trained_models_maca/` 

### Evaluating Models

```bash
# Run evaluation script
python scripts/evaluate.py

# Optional: Specify config file
python scripts/evaluate.py --config path/to/config.yaml
```

The evaluation script will:
1. Load test datasets and trained models
2. Generate predictions on test images
3. Create visualizations with bounding boxes
4. Export annotations in Label Studio format
5. Save results to:
   - `macaranga_predictions_lp3_horiz_70.png` (visualization)
   - `label_studio_format_bb_lp3_horiz_maca_70.json` (annotations)

## Configuration

The pipeline uses a schema-based configuration system. Parameters are managed through:
- `config/config.py`: Backward-compatible configuration interface
- `config/schema.py`: Configuration schema definitions
- YAML files for runtime overrides

**Before running**: You need to configure Label Studio settings by providing:
- Label Studio URL
- API key for authentication

These credentials should be set in your configuration file or environment variables before running the training or evaluation scripts.

Key configuration areas:
- **Paths**: Dataset directories, model output locations
- **Model Parameters**: SVM settings, DINOv2 model choices
- **Training Parameters**: Sample sizes, cross-validation splits
- **Species Settings**: Target species list, sensitivity targets
- **Label Studio**: URL and API key for annotation export

## Module Documentation

### data/annotation.py
- Annotation utilities for data labeling

### data/loader.py
- `DataLoader`: Handles loading TIF and PNG geospatial images
- `DataFrameLoader`: Loads and filters embedding dataframes

### data/preprocessor.py
- `ImagePreprocessor`: Image augmentation and normalization
- `DataSampler`: Balanced sampling for training

### models/dimensionality_reduction.py
- Dimensionality reduction techniques for embeddings

### models/embedding.py
- `EmbeddingGenerator`: DINOv2 feature extraction using transformers

### models/svm_trainer.py
- `SVMTrainer`: SVM training with custom biased scoring
- Supports standard F1 or biased sensitivity optimization

### models/predictor.py
- `ModelPredictor`: Batch prediction on large images
- Handles patch extraction and reconstruction

### utils/visualization.py
- `Visualizer`: Plotting utilities for images and masks
- Bounding box visualization

### utils/label_studio.py
- `LabelStudioConverter`: Export to Label Studio format
- Rectangle annotations with vote counts

### utils/geojson_converter.py
- GeoJSON format conversion utilities

### utils/morphology.py
- `MorphologyProcessor`: Binary morphological operations
- Hole filling and small object removal

### scripts/create_data.py
- Data creation and preprocessing utilities

### scripts/train.py
- Main training pipeline with argument parsing
- Multi-species training with progress tracking

### scripts/evaluate.py
- Model evaluation and prediction pipeline

## Key Technologies

- **DINOv2**: Vision transformer embeddings for feature extraction
- **SVM**: Support Vector Machine classifiers with bias correction
- **PyTorch**: Deep learning framework
- **Transformers**: HuggingFace model integration
- **Geospatial Processing**: Rasterio for TIF/PNG image handling
- **Scikit-learn**: Machine learning utilities

## Output Files

### Training
- Trained models saved in configured model directory
- Pickled model tuples (scaler, label_encoder, classifier)

### Evaluation
- Visualization images with bounding box overlays
- Label Studio JSON annotations
- GeoJSON format outputs (optional)

## Requirements

- Python >= 3.8
- CUDA-capable GPU (recommended for DINOv2)
- See `requirements.txt` for full dependencies

### Core Dependencies
- torch>=2.0.0
- transformers>=4.30.0  
- scikit-learn>=1.3.0
- rasterio>=1.3.0
- pandas>=2.0.0
- numpy>=1.24.0

## Quick Start

```bash
# Clone and install
git clone <repository-url>
cd species-detection-pipeline

# Activate conda environment with GPU support (recommended)
conda activate biased_svm

# Install dependencies (skip if already installed in conda environment)
pip install -r requirements.txt

# Train models
python scripts/train.py

# Evaluate on test data
python scripts/evaluate.py
```

**Note**: For optimal GPU performance on GPU machines, activate the `biased_svm` conda environment before running the pipeline. If the conda environment already has all dependencies installed, you can skip the `pip install -r requirements.txt` step.

## Documentation

For detailed guides and documentation, see the [docs/](docs/) directory:

- **📖 [Configuration Guide](docs/configuration-guide.md)** - Complete setup and configuration instructions
- **👨‍💻 [Developer Guide](docs/developer-guide.md)** - Architecture, coding standards, and contribution guidelines

## Contributing

When adding new features:
1. Follow existing module structure
2. Add type hints to all functions
3. Write comprehensive docstrings
4. Update configuration if needed
5. Maintain backward compatibility

## License

[Add your license here]

## Citation

If you use this pipeline in your research, please cite:

```
[Add citation information]
```

## Contact

[Add contact information]
