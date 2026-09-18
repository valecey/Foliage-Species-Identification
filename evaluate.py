"""
Evaluation script for Macaranga detection models.
Refactored from eval_macca.py with modular structure.
"""
import argparse
import gc
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from data.loader import DataLoader
from data.preprocessor import ImagePreprocessor
from models.embedding import EmbeddingGenerator
from models.predictor import ModelPredictor
from utils.visualization import Visualizer
from utils.label_studio import LabelStudioConverter
from utils.geojson_converter import GeoJSONConverter


def load_trained_models(model_dir: str, species_list: list) -> dict:
    """
    Load trained models for all species.
    
    Args:
        model_dir: Directory containing model files
        species_list: List of species names
        
    Returns:
        Dictionary mapping species names to model tuples
    """
    models = {}
    for species in species_list:
        model_path = Path(model_dir) / species
        with open(model_path, 'rb') as f:
            models[species] = pickle.load(f)
        print(f"Loaded model for: {species}")
    
    return models


def predict_and_visualize(
    image_array: np.ndarray,
    species_models: dict,
    predictor: ModelPredictor,
    config: Config
):
    """
    Generate predictions and create visualizations.
    
    Args:
        image_array: Input image array
        species_models: Dictionary of trained models
        predictor: ModelPredictor instance
        config: Configuration object
        
    Returns:
        Tuple of (figure, predictions_dict, label_studio_data)
    """
    species_list = list(species_models.keys())
    
    # Create figure
    fig, ax = Visualizer.create_figure(
        num_species=len(species_list),
        num_captures=2,
        figsize_per_row=config.FIGURE_SIZE_PER_ROW,
        wspace=config.SUBPLOT_WSPACE
    )
    
    # Storage for Label Studio format
    label_studio_data = []
    
    # Process each species
    for species_idx, species in enumerate(species_list):
        print(f"\n{'='*60}")
        print(f"Processing species {species_idx + 1}/{len(species_list)}: {species}")
        print(f"{'='*60}")
        
        # Process captures (use first test capture from config)
        test_capture = config.TEST_CAPTURES[0] if config.TEST_CAPTURES else 'test'
        capture_display_name = f"{config.DATASET_NAME.upper()}-{test_capture}"
        
        for capture_idx, capture in enumerate([capture_display_name]):
            print(f"\nProcessing capture: {capture}")
            
            # Display original image
            rgb_img = Visualizer.prepare_rgb_image(image_array)
            Visualizer.plot_image_with_predictions(
                ax[species_idx, 0],
                rgb_img,
                title=config.get_title_dict()[capture]
            )
            ax[species_idx, 0].set_ylabel(species)
            
            # Generate predictions
            print("Generating predictions...")
            img_inp = np.nan_to_num(image_array)
            clf = species_models[species]
            img_pred = predictor.batched_pred(clf, img_inp, show_progress=True)
            img_pred = (img_pred == 0).astype('uint8')
            
            print(f"Prediction shape: {img_pred.shape}")
            
            # Display mask with bounding boxes
            print("Creating visualization with bounding boxes...")
            regions = Visualizer.plot_binary_mask(
                ax[species_idx, 1],
                img_pred,
                add_bounding_boxes=True,
                box_color='red',
                box_linewidth=1
            )
            
            # Convert to Label Studio format
            print("Converting to Label Studio format...")
            annotations = LabelStudioConverter.mask_to_annotations(
                img_pred,
                species_name=species,
                species_idx=species_idx,
                capture_idx=capture_idx
            )
            
            if annotations:
                image_annotation = LabelStudioConverter.create_image_annotation(
                    species_idx=species_idx,
                    capture_idx=capture_idx,
                    annotations=annotations
                )
                label_studio_data.append(image_annotation)
            
            print(f"Found {len(annotations)} regions")
    
    # Set figure title dynamically based on configuration
    species_name = species_list[0] if species_list else "Unknown Species"
    test_capture = config.TEST_CAPTURES[0] if config.TEST_CAPTURES else "test"
    
    # Determine training dataset from model path
    training_dataset = "Unknown"
    if "casuarina" in config.SVM_MODEL_DIR.lower():
        training_dataset = "Casuarina"
    elif "lp3" in config.SVM_MODEL_DIR.lower():
        training_dataset = "LP3"
    
    # Determine evaluation dataset from config
    eval_dataset = config.DATASET_NAME.upper()
    
    plt.suptitle(
        f"Mapped {species_name} predictions on {eval_dataset} of SVMs trained on {training_dataset} dataset"
    )
    plt.tight_layout()
    
    return fig, img_pred, label_studio_data


def main():
    """Main evaluation pipeline."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate species detection models")
    parser.add_argument(
        "--config", 
        type=str, 
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="Override dataset name"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        help="Override model directory"
    )
    args = parser.parse_args()
    
    # Initialize configuration with optional overrides
    kwargs = {}
    if args.dataset_name:
        kwargs["dataset_name"] = args.dataset_name
    if args.model_dir:
        kwargs["svm_model_dir"] = args.model_dir
    
    config = Config(config_path=args.config, **kwargs)
    
    print("=" * 80)
    print("Macaranga Detection Model Evaluation")
    print("=" * 80)
    
    # Load test dataset
    print("\nLoading test dataset...")
    lp3_ds = DataLoader.load_dataset(config.LP3_M3M, config.BAND_FILENAMES_TIF)
    print(f"Loaded dataset shape: {lp3_ds.shape}")
    
    # Load LDA transformer
    print("\nLoading LDA transformer...")
    lda = torch.load(config.LDA_MODEL_PATH, weights_only=False)
    print("✓ LDA loaded")
    
    # Initialize embedding generator
    print("\nInitializing embedding generator...")
    preprocessor = ImagePreprocessor(
        crop_size=config.CROP_SIZE,
        resize_size=config.RESIZE_SIZE,
        horizontal_flip_prob=config.HORIZONTAL_FLIP_PROB
    )
    
    embedding_generator = EmbeddingGenerator(
        model_name=config.DINOV2_MODEL_NAME,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    embedding_generator.preprocessor = preprocessor
    print(f"✓ Using device: {embedding_generator.device}")
    
    # Initialize predictor
    predictor = ModelPredictor(
        embedding_generator=embedding_generator,
        lda_transformer=lda,
        patch_size=config.CROP_SIZE
    )
    
    # Load trained models
    print("\nLoading trained models...")
    species_models = load_trained_models(config.SVM_MODEL_DIR, config.SPECIES_LIST)
    
    # Garbage collection
    print("\nCleaning up memory...")
    for _ in range(5):
        gc.collect()
    
    # Generate predictions and visualizations
    print("\n" + "=" * 80)
    print("Generating Predictions")
    print("=" * 80)
    
    fig, predictions, label_studio_data = predict_and_visualize(
        image_array=lp3_ds,
        species_models=species_models,
        predictor=predictor,
        config=config
    )
    
    # Save outputs
    print("\n" + "=" * 80)
    print("Saving Outputs")
    print("=" * 80)
    
    # Save figure
    print(f"\nSaving visualization to: {config.OUTPUT_IMAGE}")
    plt.savefig(config.OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    print("✓ Visualization saved")
    
    # Save Label Studio annotations
    print(f"\nSaving Label Studio annotations to: {config.OUTPUT_JSON}")
    LabelStudioConverter.save_to_json(label_studio_data, config.OUTPUT_JSON)
    
    # Convert and save GeoJSON
    print(f"\nConverting to GeoJSON and saving to: {config.OUTPUT_GEOJSON}")
    # Try to get TIF path for automatic bounds extraction
    tif_path = None
    try:
        tif_path = str(config.LP3_M3M / config.BAND_FILENAMES_TIF[0])
        if not Path(tif_path).exists():
            tif_path = None
    except:
        tif_path = None
    
    GeoJSONConverter.convert_and_save_geojson(
        label_studio_data=label_studio_data,
        output_path=config.OUTPUT_GEOJSON,
        tif_path=tif_path
    )
    
    # Display final prediction
    print("\nDisplaying final prediction...")
    plt.figure(figsize=(10, 10))
    plt.imshow(predictions)
    plt.title("Final Prediction Mask")
    plt.axis('off')
    plt.tight_layout()
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    print(f"Outputs saved:")
    print(f"  - Visualization: {config.OUTPUT_IMAGE}")
    print(f"  - Label Studio Annotations: {config.OUTPUT_JSON}")
    print(f"  - GeoJSON: {config.OUTPUT_GEOJSON}")
    
    # Show plots
    plt.show()


if __name__ == "__main__":
    main()