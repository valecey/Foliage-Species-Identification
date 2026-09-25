"""
Configuration schema using Pydantic for validation and type safety.
"""
from pathlib import Path
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
import numpy as np


class DatasetConfig(BaseModel):
    """Dataset-specific configuration."""
    name: str = "lp3"
    train_capture: str = "first_run"
    test_captures: List[str] = ["m3m"]
    base_path: Path = Path("./lp3")
    
    @property
    def train_path(self) -> Path:
        return self.base_path / self.train_capture
    
    @property
    def test_paths(self) -> List[Path]:
        return [self.base_path / capture for capture in self.test_captures]


class ModelConfig(BaseModel):
    """Model configuration."""
    svm_model_dir: str = "lp3_first_run_trained_models_maca_v2"
    lda_model_path: str = "lda_v2_lp3_dinov2_rgb.pickle"
    dinov2_model_name: str = "facebook/dinov2-base"
    
    # Embedding parameters
    embedding_batch_size: int = 512
    embedding_num_per_crop: int = 2048
    random_patch_count: int = 65536


class DataConfig(BaseModel):
    """Data processing configuration."""
    bounds_df_path: str = "lp3_bounds_embeddings_lda_v2_dinov2_rgb.pickle"
    random_px_df_path: str = "lp3_random_embeddings_lda_v2_dinov2_rgb.pickle"
    
    # Band filenames - support both TIF and PNG formats
    band_filenames_tif: List[str] = ["result.tif"]
    band_filenames_png: List[str] = ["result.png"]
    
    # Band-specific maximum values for normalization
    band_max_values: Dict[str, Union[int, float]] = {
        "result": 2**8,           # Default RGB bands
        "result_red": 2**8,       # Red band
        "result_green": 2**8,     # Green band  
        "result_blue": 2**8,      # Blue band
        "result_nir": 2**14,      # Near Infrared (14-bit)
        "result_rededge": 2**14,  # Red Edge (14-bit)
        "rgb": 2**8,              # RGB composite
        "nir": 2**14,            # Near Infrared
        "red_edge": 2**14,        # Red Edge
        "swir": 2**16,            # Shortwave Infrared (16-bit)
        "wideband_red": 2**8,
        "wideband_green": 2**8,
        "wideband_blue": 2**8,
        "narrowband_red": 2**8,
        "narrowband_green": 2**8,
        "narrowband_blue": 2**8,
    }
    
    # Bands dictionary
    bands_dict: Dict[str, str] = {
        "rgb": "Red Green Blue",
        "wr": "Wideband Red", 
        "wg": "Wideband Green",
        "wb": "Wideband Blue",
        "r": "Narrowband Red",
        "g": "Narrowband Green", 
        "b": "Narrowband Blue",
        "nir": "Near Infrared",
        "re": "Red Edge",
        "swir": "Shortwave Infrared",
    }
    
    # Preprocessing parameters
    crop_size: int = 64
    resize_size: int = 224
    horizontal_flip_prob: float = 0.5
    
    # Normalization settings
    normalize_bands: bool = True
    auto_detect_max_values: bool = False  # If True, try to auto-detect from data
    
    # Batch processing
    batch_size: int = 1
    
    # Morphological operations
    morph_structure_size: tuple = (3, 3)
    small_holes_size: int = 10


class SpeciesConfig(BaseModel):
    """Species configuration."""
    species_list: List[str] = ["Macaranga Gigantea"]
    use_biased_svm: bool = True
    sensitivity_target: float = 0.70


class TrainingConfig(BaseModel):
    """Training hyperparameters."""
    # SVM hyperparameters (for non-biased SVM)
    gamma_values_standard: List[float] = Field(
        default_factory=lambda: [np.exp(i) for i in np.arange(-9, -2)]
    )
    c_values_standard: List[float] = Field(
        default_factory=lambda: [np.exp(i) for i in np.arange(5, 16)]
    )
    
    # SVM hyperparameters (for biased SVM)
    gamma_values_biased: List[float] = Field(
        default_factory=lambda: [np.exp(i) for i in np.arange(-5, 0)]
    )
    c_values_biased: List[float] = Field(
        default_factory=lambda: [np.exp(i) for i in np.arange(5, 12)]
    )
    w_c_values: List[float] = Field(
        default_factory=lambda: list(np.arange(0.01, 1.00, 0.01))
    )
    
    # Training configuration
    n_samples_train: int = 50_000
    n_samples_test: int = 50_000
    n_splits_cv: int = 3
    verbosity: int = 1
    n_jobs: int = 1
    random_state: int = 42


class OutputConfig(BaseModel):
    """Output configuration."""
    output_json: str = "label_studio_format_bb_lp3_horiz_maca_70.json"
    output_geojson: str = "label_studio_format_bb_lp3_horiz_maca_70.geojson"
    output_image: str = "macaranga_predictions_lp3_horiz_70.png"
    crop_output_dir: str = "crops"


class VisualizationConfig(BaseModel):
    """Visualization configuration."""
    figure_size_per_row: tuple = (10, 4.8)
    subplot_wspace: float = 0.1


class LabelStudioConfig(BaseModel):
    """Label Studio configuration."""
    url: str = "http://10.97.41.70:8080"
    api_key: str = ""
    task_first_run: int = 84
    task_m3m: int = 85


class SpeciesDetectionConfig(BaseModel):
    """Main configuration schema for species detection pipeline."""
    
    # Sub-configurations
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    species: SpeciesConfig = Field(default_factory=SpeciesConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)
    label_studio: LabelStudioConfig = Field(default_factory=LabelStudioConfig)
    
    class Config:
        arbitrary_types_allowed = True
    
    @validator("training", pre=True)
    def validate_numpy_arrays(cls, v):
        """Handle numpy array generation during validation."""
        return v
    
    # Convenience properties for backward compatibility
    @property
    def DATASET_NAME(self) -> str:
        return self.dataset.name
    
    @property
    def LP3_FIRST_RUN(self) -> Path:
        return self.dataset.train_path
    
    @property
    def LP3_M3M(self) -> Path:
        return self.dataset.test_paths[0] if self.dataset.test_paths else self.dataset.train_path
    
    @property
    def SVM_MODEL_DIR(self) -> str:
        return self.model.svm_model_dir
    
    @property
    def LDA_MODEL_PATH(self) -> str:
        return self.model.lda_model_path
    
    @property
    def BOUNDS_DF_PATH(self) -> str:
        return self.data.bounds_df_path
    
    @property
    def RANDOM_PX_DF_PATH(self) -> str:
        return self.data.random_px_df_path
    
    @property
    def BAND_FILENAMES_TIF(self) -> List[str]:
        return self.data.band_filenames_tif
    
    @property
    def BAND_FILENAMES_PNG(self) -> List[str]:
        return self.data.band_filenames_png
    
    @property
    def BANDS_DICT(self) -> Dict[str, str]:
        return self.data.bands_dict
    
    @property
    def BAND_MAX_VALUES(self) -> Dict[str, Union[int, float]]:
        return self.data.band_max_values
    
    @property
    def NORMALIZE_BANDS(self) -> bool:
        return self.data.normalize_bands
    
    @property
    def AUTO_DETECT_MAX_VALUES(self) -> bool:
        return self.data.auto_detect_max_values
    
    @property
    def SPECIES_LIST(self) -> List[str]:
        return self.species.species_list
    
    @property
    def DINOV2_MODEL_NAME(self) -> str:
        return self.model.dinov2_model_name
    
    @property
    def CROP_SIZE(self) -> int:
        return self.data.crop_size
    
    @property
    def RESIZE_SIZE(self) -> int:
        return self.data.resize_size
    
    @property
    def HORIZONTAL_FLIP_PROB(self) -> float:
        return self.data.horizontal_flip_prob
    
    @property
    def TRAIN_DATASET(self) -> str:
        return self.dataset.name
    
    @property
    def TRAIN_CAPTURE(self) -> str:
        return self.dataset.train_capture
    
    @property
    def TEST_DATASET(self) -> str:
        return self.dataset.name
    
    @property
    def TEST_CAPTURES(self) -> List[str]:
        return self.dataset.test_captures
    
    @property
    def GAMMA_VALUES_STANDARD(self) -> List[float]:
        return self.training.gamma_values_standard
    
    @property
    def C_VALUES_STANDARD(self) -> List[float]:
        return self.training.c_values_standard
    
    @property
    def GAMMA_VALUES_BIASED(self) -> List[float]:
        return self.training.gamma_values_biased
    
    @property
    def C_VALUES_BIASED(self) -> List[float]:
        return self.training.c_values_biased
    
    @property
    def W_C_VALUES(self) -> List[float]:
        return self.training.w_c_values
    
    @property
    def USE_BIASED_SVM(self) -> bool:
        return self.species.use_biased_svm
    
    @property
    def SENSITIVITY_TARGET(self) -> float:
        return self.species.sensitivity_target
    
    @property
    def N_SAMPLES_TRAIN(self) -> int:
        return self.training.n_samples_train
    
    @property
    def N_SAMPLES_TEST(self) -> int:
        return self.training.n_samples_test
    
    @property
    def N_SPLITS_CV(self) -> int:
        return self.training.n_splits_cv
    
    @property
    def VERBOSITY(self) -> int:
        return self.training.verbosity
    
    @property
    def N_JOBS(self) -> int:
        return self.training.n_jobs
    
    @property
    def RANDOM_STATE(self) -> int:
        return self.training.random_state
    
    @property
    def BATCH_SIZE(self) -> int:
        return self.data.batch_size
    
    @property
    def MORPH_STRUCTURE_SIZE(self) -> tuple:
        return self.data.morph_structure_size
    
    @property
    def SMALL_HOLES_SIZE(self) -> int:
        return self.data.small_holes_size
    
    @property
    def FIGURE_SIZE_PER_ROW(self) -> tuple:
        return self.visualization.figure_size_per_row
    
    @property
    def SUBPLOT_WSPACE(self) -> float:
        return self.visualization.subplot_wspace
    
    @property
    def OUTPUT_JSON(self) -> str:
        return self.output.output_json
    
    @property
    def OUTPUT_GEOJSON(self) -> str:
        return self.output.output_geojson
    
    @property
    def OUTPUT_IMAGE(self) -> str:
        return self.output.output_image
    
    @property
    def LABEL_STUDIO_URL(self) -> str:
        return self.label_studio.url
    
    @property
    def LABEL_STUDIO_API_KEY(self) -> str:
        return self.label_studio.api_key
    
    @property
    def LABEL_STUDIO_TASK_FIRST_RUN(self) -> int:
        return self.label_studio.task_first_run
    
    @property
    def LABEL_STUDIO_TASK_M3M(self) -> int:
        return self.label_studio.task_m3m
    
    @property
    def CROP_OUTPUT_DIR(self) -> str:
        return self.output.crop_output_dir
    
    @property
    def DINOv2_MODEL_NAME(self) -> str:
        return self.model.dinov2_model_name
    
    @property
    def EMBEDDING_BATCH_SIZE(self) -> int:
        return self.model.embedding_batch_size
    
    @property
    def EMBEDDING_NUM_PER_CROP(self) -> int:
        return self.model.embedding_num_per_crop
    
    @property
    def RANDOM_PATCH_COUNT(self) -> int:
        return self.model.random_patch_count
    
    @classmethod
    def get_alpha_values(cls, c_values: List[float]) -> List[float]:
        """Convert C values to alpha values for SGD."""
        return [1/i for i in c_values]
    
    @classmethod
    def get_class_weights(cls, w_c_values: List[float]) -> List[Dict[int, float]]:
        """Generate class weight configurations."""
        return [{0: i, 1: 1-i} for i in w_c_values]
    
    def get_title_dict(self) -> Dict[str, str]:
        """Get capture type to title mapping."""
        test_capture = self.TEST_CAPTURES[0] if self.TEST_CAPTURES else 'test'
        capture_key = f"{self.DATASET_NAME.upper()}-{test_capture}"
        return {capture_key: f'{self.DATASET_NAME.upper()} {test_capture} (Test)'}
