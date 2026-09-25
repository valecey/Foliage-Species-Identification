# Configuration Guide

This guide provides detailed instructions for configuring and running the species detection pipeline.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Configuration Files](#configuration-files)
4. [Label Studio Setup](#label-studio-setup)
5. [Data Configuration](#data-configuration)
6. [Running the Pipeline](#running-the-pipeline)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements
- Python >= 3.8
- CUDA-capable GPU (recommended for DINOv2)
- Sufficient disk space for datasets and models
- Git for cloning the repository

### Required Software
- Conda/Miniconda
- Git
- Access to Label Studio (if using annotation features)

## Environment Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd species-detection-pipeline
```

### 2. Activate Conda Environment
```bash
# Activate the pre-configured environment
conda activate biased_svm
```

### 3. Install Dependencies (if needed)
```bash
# Only if dependencies aren't already installed in the conda environment
pip install -r requirements.txt
```

### 4. Verify GPU Access
```bash
# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Configuration Files

### Main Configuration Structure
The pipeline uses a schema-based configuration system with Pydantic validation:

```
config/
├── config.py          # Main configuration interface (backward-compatible)
├── schema.py          # Configuration schema definitions with Pydantic models
├── loader.py          # Configuration loading and merging utilities
└── default_config.yaml # Default configuration values
```

### Configuration Schema
The configuration is organized into logical sections using Pydantic models:

#### 1. Dataset Configuration (`DatasetConfig`)
```yaml
dataset:
  name: "lp3"                    # Dataset identifier
  train_capture: "first_run"      # Training capture name
  test_captures: ["m3m"]         # List of test captures
  base_path: "./lp3"              # Base dataset directory
```

#### 2. Model Configuration (`ModelConfig`)
```yaml
model:
  svm_model_dir: "lp3_first_run_trained_models_maca_v2"
  lda_model_path: "lda_v2_lp3_dinov2_rgb.pickle"
  dinov2_model_name: "facebook/dinov2-base"
  embedding_batch_size: 512
  embedding_num_per_crop: 2048
  random_patch_count: 65536
```

#### 3. Data Processing Configuration (`DataConfig`)
```yaml
data:
  bounds_df_path: "lp3_bounds_embeddings_lda_v2_dinov2_rgb.pickle"
  random_px_df_path: "lp3_random_embeddings_lda_v2_dinov2_rgb.pickle"
  band_filenames_tif: ["result.tif"]
  band_filenames_png: ["result.png"]
  bands_dict:
    rgb: "Red Green Blue"
    wr: "Wideband Red"
    wg: "Wideband Green"
    wb: "Wideband Blue"
  crop_size: 64
  resize_size: 224
  horizontal_flip_prob: 0.5
  batch_size: 1
  morph_structure_size: [3, 3]
  small_holes_size: 10
```

#### 4. Species Configuration (`SpeciesConfig`)
```yaml
species:
  species_list: ["Macaranga Gigantea"]
  use_biased_svm: true
  sensitivity_target: 0.70
```

#### 5. Training Configuration (`TrainingConfig`)
```yaml
training:
  # SVM hyperparameters for standard SVM
  gamma_values_standard: [0.0001, 0.001, 0.01, 0.1, 1.0]
  c_values_standard: [148.4, 403.4, 1096.6, 2980.9, 8103.1, 22026.5]
  
  # SVM hyperparameters for biased SVM
  gamma_values_biased: [0.0067, 0.0183, 0.0498, 0.1353, 0.3679]
  c_values_biased: [148.4, 403.4, 1096.6, 2980.9, 8103.1]
  w_c_values: [0.01, 0.02, 0.03, ..., 0.99]  # 0.01 to 0.99
  
  # Training parameters
  n_samples_train: 50000
  n_samples_test: 50000
  n_splits_cv: 3
  verbosity: 1
  n_jobs: 1
  random_state: 42
```

#### 6. Output Configuration (`OutputConfig`)
```yaml
output:
  output_json: "label_studio_format_bb_lp3_horiz_maca_70.json"
  output_geojson: "label_studio_format_bb_lp3_horiz_maca_70.geojson"
  output_image: "macaranga_predictions_lp3_horiz_70.png"
  crop_output_dir: "crops"
```

#### 7. Visualization Configuration (`VisualizationConfig`)
```yaml
visualization:
  figure_size_per_row: [10, 4.8]
  subplot_wspace: 0.1
```

#### 8. Label Studio Configuration (`LabelStudioConfig`)
```yaml
label_studio:
  url: "http://your-label-studio-instance:8080"
  api_key: "your-api-key-here"
  task_first_run: 84   # Training task ID
  task_m3m: 85         # Test task ID
```

### Configuration Loading and Merging

The configuration system supports:
1. **Default values**: Built-in defaults from schema
2. **YAML overrides**: Override defaults with YAML files
3. **Runtime overrides**: Override with keyword arguments
4. **Environment variables**: Override with environment variables

#### Loading Configuration
```python
from config.config import Config

# Load with YAML file
config = Config(config_path="path/to/config.yaml")

# Load with runtime overrides
config = Config(dataset_name="custom_dataset", species_list=["species1"])

# Load with both
config = Config(
    config_path="path/to/config.yaml",
    dataset_name="override_name"
)
```

#### Configuration Merging Priority
1. Default values (lowest priority)
2. YAML configuration file
3. Runtime keyword arguments (highest priority)

#### Dot Notation Support
```python
# Override nested configuration using dot notation
config = Config(
    "dataset.name": "custom_dataset",
    "model.svm_model_dir": "custom_models",
    "label_studio.task_first_run": 100
)
```

### Backward Compatibility

The configuration system maintains backward compatibility through property aliases:

```python
# Old-style access (still works)
config.DATASET_NAME
config.LP3_FIRST_RUN
config.SVM_MODEL_DIR
config.SPECIES_LIST

# New-style access
config.dataset.name
config.dataset.train_path
config.model.svm_model_dir
config.species.species_list
```

### Configuration Validation

All configurations are validated using Pydantic:
- Type checking
- Value constraints
- Required field validation
- Custom validation rules

```python
from config.schema import SpeciesDetectionConfig

# This will raise ValidationError if invalid
config = SpeciesDetectionConfig(
    dataset=DatasetConfig(name="invalid_dataset"),  # Will be validated
    species=SpeciesConfig(sensitivity_target=1.5)     # Will raise error (> 1.0)
)
```

## Label Studio Setup

### 1. Obtain Label Studio Credentials
- **URL**: Your Label Studio instance URL (e.g., `http://localhost:8080`)
- **API Key**: Generate from Label Studio interface (Account → Settings → API Key)

### 2. Configure Label Studio Settings
Add to your configuration file or set as environment variables:

```yaml
label_studio:
  url: "http://your-label-studio-instance:8080"
  api_key: "your-api-key-here"
  task_first_run: 84   # Training task ID (e.g., LP3 first run task)
  task_m3m: 85         # Test/evaluation task ID (e.g., LP3 m3m task)
```

**Note**: The pipeline uses task IDs instead of project IDs to retrieve specific annotation data. You need to provide the task IDs for:
- `task_first_run`: Training data task
- `task_m3m`: Test/evaluation data task

**Examples from actual configuration files**:
- For LP3 dataset: `task_first_run: 84`, `task_m3m: 85`
- For Casuarina dataset: `task_first_run: 3`, `task_m3m: 2`

Or as environment variables:
```bash
export LABEL_STUDIO_URL="http://your-label-studio-instance:8080"
export LABEL_STUDIO_API_KEY="your-api-key-here"
export LABEL_STUDIO_TASK_FIRST_RUN=84
export LABEL_STUDIO_TASK_M3M=85
```

### 3. Test Label Studio Connection
```python
from utils.label_studio import LabelStudioConverter

# Test connection
converter = LabelStudioConverter()
# This will raise an error if credentials are invalid
```

## Data Configuration

### Dataset Structure
Your datasets should follow this structure:

```
datasets/
├── training/
│   ├── images/
│   │   ├── image1.tif
│   │   ├── image2.tif
│   │   └── ...
│   └── annotations/
│       ├── bounds.csv
│       └── random_pixels.csv
└── test/
    ├── images/
    └── annotations/
```

### Band Files Configuration
```yaml
data:
  band_filenames_png: ["band_R.png", "band_G.png", "band_B.png"]
  # Or for TIF files
  band_filenames_tif: ["band_R.tif", "band_G.tif", "band_B.tif"]
```

### Embedding DataFrames
The pipeline expects two main dataframes:

1. **bounds_df**: Contains species bounding box information
   - Columns: `name`, `x`, `y`, `width`, `height`, `capture`, plus embedding columns

2. **random_px_df**: Contains random pixel samples for negative class
   - Columns: `x`, `y`, `capture`, plus embedding columns

## Running the Pipeline

### 1. Create Data (Optional)
```bash
# If you need to create or preprocess data
python scripts/create_data.py --config path/to/config.yaml
```

### 2. Train Models
```bash
# Basic training
python scripts/train.py

# With custom configuration
python scripts/train.py --config path/to/config.yaml

# Override specific parameters
python scripts/train.py --species-list species1 species2 --dataset-name custom_dataset
```

### 3. Evaluate Models
```bash
# Basic evaluation
python scripts/evaluate.py

# With custom configuration
python scripts/evaluate.py --config path/to/config.yaml
```

### 4. Monitor Progress
Training and evaluation scripts provide progress bars and detailed logging:
- Training progress per species
- Model performance metrics
- File output locations

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size or use CPU
export CUDA_VISIBLE_DEVICES=""  # Force CPU usage
```

#### 2. Missing Dependencies
```bash
# Reinstall with specific versions
pip install -r requirements.txt --force-reinstall
```

#### 3. Label Studio Connection Issues
```bash
# Verify URL and API key
curl -H "Authorization: Token your-api-key" http://your-label-studio-instance:8080/api/projects
```

#### 4. Dataset Path Issues
```bash
# Verify paths exist and are accessible
ls -la /path/to/your/dataset
```

### Debug Mode
Enable verbose logging for debugging:
```yaml
logging:
  verbosity: 2  # 0=quiet, 1=normal, 2=verbose
```

### Performance Optimization

#### GPU Utilization
```bash
# Monitor GPU usage
nvidia-smi

# Set GPU memory growth
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

#### Parallel Processing
```yaml
training:
  n_jobs: -1  # Use all available cores
```

## Configuration Validation

### Validate Configuration
```python
from config.config import Config

# Load and validate configuration
config = Config(config_path="path/to/config.yaml")

# Check key parameters
print(f"Species list: {config.SPECIES_LIST}")
print(f"Model directory: {config.SVM_MODEL_DIR}")
print(f"Using biased SVM: {config.USE_BIASED_SVM}")
```

### Test Data Loading
```python
from data.loader import DataLoader, DataFrameLoader

# Test dataset loading
dataset = DataLoader.load_dataset(config.LP3_FIRST_RUN, config.BAND_FILENAMES_PNG)
print(f"Dataset shape: {dataset.shape}")

# Test embedding loading
bounds_df, random_px_df = DataFrameLoader.load_embeddings(
    config.BOUNDS_DF_PATH, config.RANDOM_PX_DF_PATH
)
print(f"Bounds shape: {bounds_df.shape}")
print(f"Random pixels shape: {random_px_df.shape}")
```

## Next Steps

After successful configuration:
1. Run training with a small subset first
2. Validate model outputs
3. Scale up to full dataset
4. Monitor training progress
5. Evaluate and analyze results

For maintenance and development guidance, see:
- [Maintenance Guide](maintenance-guide.md)
- [Developer Guide](developer-guide.md)
