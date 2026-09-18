# Developer Guide

This guide provides comprehensive information for developers working on the species detection pipeline, including architecture, coding standards, testing, and contribution guidelines.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Development Setup](#development-setup)
3. [Code Structure](#code-structure)
4. [Coding Standards](#coding-standards)
5. [Testing](#testing)
6. [Adding New Features](#adding-new-features)
7. [Debugging](#debugging)
8. [Performance Optimization](#performance-optimization)
9. [Documentation](#documentation)
10. [Contribution Guidelines](#contribution-guidelines)

## Architecture Overview

### System Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │   Model Layer   │    │  Utility Layer  │
│                 │    │                 │    │                 │
│ • DataLoader    │───▶│ • EmbeddingGen  │───▶│ • Visualizer    │
│ • Preprocessor  │    │ • SVMTrainer    │    │ • Morphology    │
│ • DataSampler   │    │ • Predictor     │    │ • Converters    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Configuration   │
                    │                 │
                    │ • Config        │
                    │ • Schema        │
                    │ • Loader        │
                    └─────────────────┘
```

### Data Flow
1. **Input**: Geospatial images (TIF/PNG) + annotation data
2. **Processing**: DINOv2 embedding generation → SVM training → prediction
3. **Output**: Species predictions with bounding boxes

### Key Components

#### Configuration System
- **Schema-based**: Type-safe configuration with validation
- **Hierarchical**: YAML overrides + programmatic configuration
- **Backward-compatible**: Maintains legacy interface

#### Model Pipeline
- **Embedding Generation**: DINOv2 vision transformer features
- **SVM Training**: Biased SVM with sensitivity optimization
- **Batch Prediction**: Efficient processing of large images

#### Data Handling
- **Multi-format**: TIF and PNG image support
- **Balanced Sampling**: Automatic class balancing
- **Geospatial**: Coordinate system handling

## Development Setup

### 1. Clone and Setup
```bash
git clone <repository-url>
cd species-detection-pipeline

# Create development environment
conda create -n species_detection_dev python=3.9
conda activate species_detection_dev

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # If exists
```

### 2. Development Dependencies
Create `requirements-dev.txt`:
```
# Testing
pytest>=7.0.0
pytest-cov>=4.0.0
pytest-mock>=3.10.0

# Code quality
black>=23.0.0
isort>=5.12.0
flake8>=6.0.0
mypy>=1.0.0

# Documentation
sphinx>=6.0.0
sphinx-rtd-theme>=1.2.0

# Development tools
pre-commit>=3.0.0
jupyter>=1.0.0
```

### 3. Pre-commit Hooks
```bash
# Install pre-commit
pre-commit install

# Create .pre-commit-config.yaml
cat > .pre-commit-config.yaml << 'EOF'
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
EOF
```

### 4. IDE Configuration

#### VS Code Settings
Create `.vscode/settings.json`:
```json
{
    "python.defaultInterpreterPath": "/path/to/conda/envs/species_detection_dev/bin/python",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    }
}
```

## Code Structure

### Module Organization
```
species-detection-pipeline/
├── config/                 # Configuration management
│   ├── __init__.py
│   ├── config.py           # Main configuration interface
│   ├── schema.py           # Configuration schema
│   └── loader.py           # Configuration loading
├── data/                   # Data handling
│   ├── __init__.py
│   ├── annotation.py       # Annotation utilities
│   ├── loader.py           # Data loading
│   └── preprocessor.py     # Data preprocessing
├── models/                 # Machine learning models
│   ├── __init__.py
│   ├── dimensionality_reduction.py
│   ├── embedding.py        # DINOv2 embeddings
│   ├── predictor.py        # Model prediction
│   └── svm_trainer.py      # SVM training
├── utils/                  # Utilities
│   ├── __init__.py
│   ├── geojson_converter.py
│   ├── label_studio.py
│   ├── morphology.py
│   └── visualization.py
├── scripts/                # Entry points
│   ├── __init__.py
│   ├── create_data.py
│   ├── evaluate.py
│   └── train.py
└── tests/                  # Tests
    ├── __init__.py
    ├── test_config/
    ├── test_data/
    ├── test_models/
    └── test_utils/
```

### Design Patterns

#### Configuration Pattern
```python
# config/schema.py
from pydantic import BaseModel, Field
from typing import List, Optional

class ModelConfig(BaseModel):
    use_biased_svm: bool = True
    sensitivity_target: float = Field(0.70, ge=0.0, le=1.0)
    gamma_values: List[float] = [0.001, 0.01, 0.1, 1.0]
    c_values: List[float] = [0.1, 1.0, 10.0, 100.0]

class SpeciesDetectionConfig(BaseModel):
    model: ModelConfig = ModelConfig()
    # ... other config sections
```

#### Factory Pattern for Models
```python
# models/factory.py
from typing import Dict, Type
from .embedding import EmbeddingGenerator
from .svm_trainer import SVMTrainer

class ModelFactory:
    _models: Dict[str, Type] = {
        'embedding': EmbeddingGenerator,
        'svm': SVMTrainer,
    }
    
    @classmethod
    def create(cls, model_type: str, **kwargs):
        if model_type not in cls._models:
            raise ValueError(f"Unknown model type: {model_type}")
        return cls._models[model_type](**kwargs)
```

#### Strategy Pattern for Data Processing
```python
# data/strategies.py
from abc import ABC, abstractmethod

class ProcessingStrategy(ABC):
    @abstractmethod
    def process(self, data):
        pass

class TIFProcessingStrategy(ProcessingStrategy):
    def process(self, data):
        # TIF-specific processing
        pass

class PNGProcessingStrategy(ProcessingStrategy):
    def process(self, data):
        # PNG-specific processing
        pass
```

## Coding Standards

### Python Style Guide
- **Formatter**: Black with default settings
- **Import sorting**: isort with black profile
- **Linting**: flake8 with strict rules
- **Type checking**: mypy with strict mode

### Code Formatting
```python
# Use Black formatting
# Line length: 88 characters
# Double quotes for strings
# Trailing commas in multi-line constructs

# Example:
def process_data(
    data: pd.DataFrame,
    config: Config,
    verbose: bool = False,
) -> pd.DataFrame:
    """Process input data according to configuration.
    
    Args:
        data: Input dataframe to process
        config: Configuration object
        verbose: Enable verbose logging
        
    Returns:
        Processed dataframe
    """
    if verbose:
        print("Starting data processing...")
    
    # Processing logic here
    processed_data = data.copy()
    
    return processed_data
```

### Type Hints
```python
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    hyperparameters: Dict[str, Union[float, int]],
    validation_split: float = 0.2,
) -> Tuple[object, Dict[str, float]]:
    """Train a model with given hyperparameters.
    
    Returns:
        Tuple of (trained_model, metrics_dict)
    """
    pass
```

### Documentation Standards
```python
def complex_function(
    param1: str,
    param2: Optional[int] = None,
    **kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    """Brief description of the function.
    
    Detailed description explaining the function's purpose,
    algorithm, and any important considerations.
    
    Args:
        param1: Description of first parameter
        param2: Description of second parameter
        
    Keyword Args:
        Additional keyword arguments for customization
        
    Returns:
        Dictionary containing results with keys:
        - 'result': Main result value
        - 'metadata': Additional information
        
    Raises:
        ValueError: If param1 is invalid
        TypeError: If param2 is not an integer
        
    Example:
        >>> result = complex_function("test", param2=42)
        >>> print(result['result'])
        'success'
    """
    pass
```

### Error Handling
```python
# Custom exceptions
class SpeciesDetectionError(Exception):
    """Base exception for species detection pipeline."""
    pass

class ConfigurationError(SpeciesDetectionError):
    """Raised when configuration is invalid."""
    pass

class DataLoadError(SpeciesDetectionError):
    """Raised when data loading fails."""
    pass

# Usage
def load_config(config_path: str) -> Config:
    """Load configuration from file."""
    try:
        if not Path(config_path).exists():
            raise ConfigurationError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return Config(**config_data)
        
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in config file: {e}")
    except Exception as e:
        raise ConfigurationError(f"Failed to load config: {e}")
```

## Testing

### Test Structure
```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── test_config/
│   ├── test_config.py
│   └── test_schema.py
├── test_data/
│   ├── test_loader.py
│   └── test_preprocessor.py
├── test_models/
│   ├── test_embedding.py
│   ├── test_svm_trainer.py
│   └── test_predictor.py
└── test_utils/
    ├── test_visualization.py
    └── test_converters.py
```

### Test Fixtures
```python
# tests/conftest.py
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

@pytest.fixture
def sample_config():
    """Provide a sample configuration for testing."""
    return Config(
        model=ModelConfig(
            use_biased_svm=True,
            sensitivity_target=0.70
        )
    )

@pytest.fixture
def sample_data():
    """Provide sample data for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        'name': ['species1', 'species2', 'species1', 'species2'],
        'x': np.random.rand(4),
        'y': np.random.rand(4),
        0: np.random.rand(4),  # Embedding column
        1: np.random.rand(4),  # Embedding column
    })

@pytest.fixture
def temp_dir(tmp_path):
    """Provide a temporary directory for testing."""
    return tmp_path
```

### Unit Tests
```python
# tests/test_models/test_svm_trainer.py
import pytest
import numpy as np
from models.svm_trainer import SVMTrainer
from config.config import Config

class TestSVMTrainer:
    def test_initialization(self, sample_config):
        """Test SVM trainer initialization."""
        trainer = SVMTrainer(
            gamma_values=[0.01, 0.1],
            c_values=[1.0, 10.0],
            use_biased_svm=True,
            sensitivity_target=0.70
        )
        
        assert trainer.gamma_values == [0.01, 0.1]
        assert trainer.c_values == [1.0, 10.0]
        assert trainer.use_biased_svm is True
    
    def test_training_with_valid_data(self, sample_data):
        """Test training with valid input data."""
        trainer = SVMTrainer(
            gamma_values=[0.01],
            c_values=[1.0],
            use_biased_svm=False
        )
        
        X = sample_data[[0, 1]].to_numpy()
        y = sample_data['name'].to_numpy()
        
        scaler, label_encoder, model = trainer.train(X, y, X, y, "test_species")
        
        assert scaler is not None
        assert label_encoder is not None
        assert model is not None
    
    def test_training_with_invalid_data(self):
        """Test training with invalid input data."""
        trainer = SVMTrainer()
        
        with pytest.raises(ValueError):
            trainer.train(np.array([]), np.array([]), np.array([]), np.array([]), "test")
```

### Integration Tests
```python
# tests/test_integration.py
import pytest
from pathlib import Path
from scripts.train import main as train_main
from scripts.evaluate import main as eval_main

class TestIntegration:
    def test_full_pipeline(self, temp_dir, sample_config):
        """Test the complete training and evaluation pipeline."""
        # Setup test data
        test_data_dir = temp_dir / "test_data"
        test_data_dir.mkdir()
        
        # Create minimal test dataset
        # ... setup code ...
        
        # Run training
        with pytest.raises(SystemExit) as exc_info:
            train_main()
        
        assert exc_info.value.code == 0
        
        # Run evaluation
        with pytest.raises(SystemExit) as exc_info:
            eval_main()
        
        assert exc_info.value.code == 0
```

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=species_detection --cov-report=html

# Run specific test file
pytest tests/test_models/test_svm_trainer.py

# Run with verbose output
pytest -v

# Run tests matching pattern
pytest -k "test_training"
```

## Adding New Features

### 1. Adding a New Model Type

#### Step 1: Create Model Class
```python
# models/new_model.py
from abc import ABC, abstractmethod
import numpy as np

class NewModelTrainer(ABC):
    """New model trainer implementation."""
    
    def __init__(self, **kwargs):
        self.params = kwargs
    
    def train(self, X: np.ndarray, y: np.ndarray) -> object:
        """Train the new model."""
        # Implementation here
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        # Implementation here
        pass
```

#### Step 2: Update Configuration
```python
# config/schema.py
class NewModelConfig(BaseModel):
    param1: float = 1.0
    param2: int = 100
    enabled: bool = False

class SpeciesDetectionConfig(BaseModel):
    # ... existing configs ...
    new_model: NewModelConfig = NewModelConfig()
```

#### Step 3: Add Factory Support
```python
# models/factory.py
class ModelFactory:
    _models = {
        'embedding': EmbeddingGenerator,
        'svm': SVMTrainer,
        'new_model': NewModelTrainer,  # Add new model
    }
```

#### Step 4: Write Tests
```python
# tests/test_models/test_new_model.py
import pytest
from models.new_model import NewModelTrainer

class TestNewModelTrainer:
    def test_initialization(self):
        trainer = NewModelTrainer(param1=2.0, param2=200)
        assert trainer.params['param1'] == 2.0
        assert trainer.params['param2'] == 200
    
    def test_training(self, sample_data):
        trainer = NewModelTrainer()
        X = sample_data[[0, 1]].to_numpy()
        y = sample_data['name'].to_numpy()
        
        model = trainer.train(X, y)
        assert model is not None
```

### 2. Adding New Data Format Support

#### Step 1: Create Data Loader
```python
# data/new_format_loader.py
class NewFormatLoader:
    """Loader for new data format."""
    
    @staticmethod
    def load(file_path: str) -> pd.DataFrame:
        """Load data from new format."""
        # Implementation here
        pass
```

#### Step 2: Update Factory
```python
# data/loader.py
class DataLoader:
    _loaders = {
        'tif': TIFLoader,
        'png': PNGLoader,
        'new_format': NewFormatLoader,  # Add new loader
    }
    
    @classmethod
    def load_dataset(cls, path: str, format_type: str):
        if format_type not in cls._loaders:
            raise ValueError(f"Unsupported format: {format_type}")
        
        return cls._loaders[format_type].load(path)
```

## Debugging

### Debugging Tools

#### 1. Logging Configuration
```python
# utils/debug.py
import logging
from typing import Optional

def setup_debug_logging(
    level: str = "DEBUG",
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
) -> None:
    """Setup debugging logging configuration."""
    
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_string,
        handlers=handlers,
    )
```

#### 2. Memory Profiling
```python
# utils/profiling.py
import tracemalloc
import time
from functools import wraps
from typing import Callable

def profile_memory(func: Callable) -> Callable:
    """Decorator to profile memory usage of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        tracemalloc.start()
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        elapsed_time = time.time() - start_time
        
        print(f"{func.__name__}:")
        print(f"  Time: {elapsed_time:.2f}s")
        print(f"  Memory usage: {current / 1024 / 1024:.1f}MB (current)")
        print(f"  Peak memory: {peak / 1024 / 1024:.1f}MB")
        
        return result
    return wrapper
```

#### 3. Data Validation
```python
# utils/validation.py
import pandas as pd
import numpy as np
from typing import Any, Dict, List

def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> None:
    """Validate dataframe has required columns and no null values."""
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    null_columns = df.columns[df.isnull().any()].tolist()
    if null_columns:
        raise ValueError(f"Columns contain null values: {null_columns}")

def validate_embeddings(embeddings: np.ndarray, expected_dim: int) -> None:
    """Validate embedding dimensions."""
    if len(embeddings.shape) != 2:
        raise ValueError(f"Expected 2D embeddings, got {len(embeddings.shape)}D")
    
    if embeddings.shape[1] != expected_dim:
        raise ValueError(f"Expected {expected_dim} dimensions, got {embeddings.shape[1]}")
```

### Common Debugging Scenarios

#### 1. Model Training Issues
```python
# Debug SVM training
def debug_svm_training(X, y, config):
    """Debug SVM training process."""
    print(f"Input shape: {X.shape}")
    print(f"Label distribution: {pd.Series(y).value_counts()}")
    
    # Check for data issues
    if np.any(np.isnan(X)):
        print("Warning: NaN values found in features")
    
    if np.any(np.isinf(X)):
        print("Warning: Infinite values found in features")
    
    # Check class balance
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"Class balance: {dict(zip(unique_labels, counts))}")
```

#### 2. Memory Issues
```python
# Debug memory usage
def debug_memory_usage():
    """Debug current memory usage."""
    import psutil
    import torch
    
    # System memory
    memory = psutil.virtual_memory()
    print(f"System memory: {memory.percent}% used")
    
    # GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU memory allocated: {gpu_memory:.1f}GB")
```

## Performance Optimization

### 1. Vectorization
```python
# Bad: Slow loop-based processing
def slow_processing(features):
    results = []
    for i in range(len(features)):
        result = complex_calculation(features[i])
        results.append(result)
    return results

# Good: Vectorized processing
def fast_processing(features):
    return np.vectorize(complex_calculation)(features)
```

### 2. Memory Optimization
```python
# Use generators for large datasets
def process_large_dataset(file_path):
    """Process large dataset without loading everything into memory."""
    for chunk in pd.read_csv(file_path, chunksize=1000):
        yield process_chunk(chunk)

# Use memory mapping for large arrays
def load_large_array(file_path):
    """Load large array using memory mapping."""
    return np.memmap(file_path, dtype='float32', mode='r')
```

### 3. Parallel Processing
```python
from multiprocessing import Pool
from functools import partial

def parallel_process(data, func, n_workers=4):
    """Process data in parallel."""
    with Pool(n_workers) as pool:
        results = pool.map(func, data)
    return results
```

## Documentation

### 1. Code Documentation
- All public functions must have docstrings
- Use Google or NumPy docstring format
- Include type hints for all functions
- Document complex algorithms with inline comments

### 2. API Documentation
Generate API docs with Sphinx:
```bash
# Install Sphinx
pip install sphinx sphinx-rtd-theme

# Initialize docs
sphinx-quickstart docs/

# Generate API docs
sphinx-apidoc -o docs/source/ species_detection/

# Build HTML docs
cd docs
make html
```

### 3. Example Documentation
Create `examples/` directory with:
- Basic usage examples
- Advanced configuration examples
- Custom model integration examples
- Performance optimization examples

## Contribution Guidelines

### 1. Development Workflow
```bash
# 1. Create feature branch
git checkout -b feature/new-feature

# 2. Make changes with proper commits
git add .
git commit -m "feat: add new feature description"

# 3. Run tests
pytest

# 4. Run code quality checks
black .
isort .
flake8 .
mypy .

# 5. Push and create PR
git push origin feature/new-feature
```

### 2. Commit Message Format
```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Code style
- `refactor`: Code refactoring
- `test`: Test addition/modification
- `chore`: Maintenance

### 3. Pull Request Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
```

### 4. Code Review Process
1. All PRs require at least one approval
2. Automated tests must pass
3. Code quality checks must pass
4. Documentation must be updated for API changes
5. Breaking changes require discussion and approval

This developer guide provides comprehensive information for contributing to the species detection pipeline. Following these guidelines ensures code quality, maintainability, and collaboration efficiency.
