"""
Data creation script for species detection pipeline.
Converts raw images and annotations to training embeddings.
Reproduces the exact logic from create_maca.py in modular form.
"""
import argparse
import os
import sys
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from data.loader import DataLoader
from data.annotation import AnnotationFetcher, BoundsProcessor
from models.embedding import EmbeddingGenerator
from models.dimensionality_reduction import DataCreationPipeline


def fetch_annotations(config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch annotations from Label Studio for both datasets.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_bounds, test_bounds)
    """
    print("=" * 80)
    print("Fetching Annotations from Label Studio")
    print("=" * 80)
    
    # Initialize annotation fetcher
    fetcher = AnnotationFetcher(config)
    
    # Fetch train annotations
    print("Fetching train annotations...")
    train_data = fetcher.fetch_annotation(
        config.LABEL_STUDIO_TASK_FIRST_RUN,
        f'{config.DATASET_NAME}_annotation_task_{config.TRAIN_CAPTURE}.json'
    )
    train_bounds = fetcher.process_annotations_to_csv(
        train_data,
        str(config.LP3_FIRST_RUN / 'bounds.csv')
    )
    
    # Fetch test annotations
    print("Fetching test annotations...")
    test_data = fetcher.fetch_annotation(
        config.LABEL_STUDIO_TASK_M3M,
        f'{config.DATASET_NAME}_annotation_task_{config.TEST_CAPTURES[0]}.json'
    )
    test_bounds = fetcher.process_annotations_to_csv(
        test_data,
        str(config.LP3_M3M / 'bounds.csv')
    )
    
    return train_bounds, test_bounds


def process_bounds_data(
    train_bounds: pd.DataFrame,
    test_bounds: pd.DataFrame,
    config: Config
) -> pd.DataFrame:
    """
    Process bounds data with metadata and crops.
    Now properly combines both train and test datasets.
    
    Args:
        train_bounds: Train bounds dataframe
        test_bounds: Test bounds dataframe
        config: Configuration object
        
    Returns:
        Combined bounds dataframe with crops
    """
    print("\n" + "=" * 80)
    print("Processing Bounds Data")
    print("=" * 80)
    
    # Initialize bounds processor
    processor = BoundsProcessor()
    
    # Add metadata to train bounds
    print("Adding metadata to train bounds...")
    train_bounds = processor.add_metadata(
        train_bounds, config.TRAIN_DATASET, config.TRAIN_CAPTURE
    )
    
    # Add metadata to test bounds
    print("Adding metadata to test bounds...")
    test_bounds = processor.add_metadata(
        test_bounds, config.TEST_DATASET, config.TEST_CAPTURES[0]
    )
    
    # Load datasets
    print("Loading image datasets...")
    train_ds = DataLoader.load_dataset(
        config.LP3_FIRST_RUN, config.BAND_FILENAMES_PNG
    )
    test_ds = DataLoader.load_dataset(
        config.LP3_M3M, config.BAND_FILENAMES_PNG
    )
    
    # Add crops to both datasets
    print("Extracting crops from train data...")
    if len(train_bounds) > 0:
        train_bounds = processor.add_crops_to_dataframe(train_bounds, train_ds)
    
    print("Extracting crops from test data...")
    if len(test_bounds) > 0:
        test_bounds = processor.add_crops_to_dataframe(test_bounds, test_ds)
    
    # Combine both datasets (matching original logic)
    bounds_df = pd.concat([train_bounds, test_bounds], ignore_index=True)
    print(f"Combined bounds dataframe: {len(bounds_df)} crops")
    print(f"  - Train: {len(train_bounds)} crops")
    print(f"  - Test: {len(test_bounds)} crops")
    
    return bounds_df


def generate_embeddings(
    bounds_df: pd.DataFrame,
    config: Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate embeddings for bounds and random patches.
    Matches original logic exactly.
    
    Args:
        bounds_df: Bounds dataframe with crops
        config: Configuration object
        
    Returns:
        Tuple of (bounds_embeddings_df, random_embeddings_df)
    """
    print("\n" + "=" * 80)
    print("Generating Embeddings")
    print("=" * 80)
    
    # Load datasets for random embeddings
    print("Loading image datasets for random embeddings...")
    train_ds = DataLoader.load_dataset(
        config.LP3_FIRST_RUN, config.BAND_FILENAMES_PNG
    )
    test_ds = DataLoader.load_dataset(
        config.LP3_M3M, config.BAND_FILENAMES_PNG
    )
    
    # Initialize embedding generator
    embedder = EmbeddingGenerator(model_name=config.DINOv2_MODEL_NAME)
    
    # Save crops to disk (optional, matches original behavior)
    print("Saving crops to disk...")
    embedder.save_crops_to_disk(bounds_df, config.CROP_OUTPUT_DIR)
    
    # Generate bounds embeddings
    print("Generating bounds embeddings...")
    bounds_embeddings_df = embedder.generate_bounds_embeddings(
        bounds_df,
        crop_size=config.CROP_SIZE,
        resize_size=config.RESIZE_SIZE,
        horizontal_flip_prob=config.HORIZONTAL_FLIP_PROB,
        num_per_crop=config.EMBEDDING_NUM_PER_CROP,
        batch_size=config.EMBEDDING_BATCH_SIZE
    )
    
    # Generate random patch embeddings (matching original exactly)
    print("Generating random patch embeddings...")
    print("Train random patches...")
    train_random_df = embedder.generate_random_patch_embeddings(
        train_ds,
        n_patches=config.RANDOM_PATCH_COUNT,
        batch_size=config.EMBEDDING_BATCH_SIZE
    )
    
    print("Test random patches...")
    test_random_df = embedder.generate_random_patch_embeddings(
        test_ds,
        n_patches=config.RANDOM_PATCH_COUNT,
        batch_size=config.EMBEDDING_BATCH_SIZE
    )
    
    # Add metadata to random embeddings (matching original)
    train_random_df['ds'] = config.TRAIN_DATASET
    train_random_df['capture'] = config.TRAIN_CAPTURE
    test_random_df['ds'] = config.TEST_DATASET
    test_random_df['capture'] = config.TEST_CAPTURES[0]
    
    # Combine random embeddings (matching original)
    random_embeddings_df = pd.concat([train_random_df, test_random_df])
    
    print(f"Bounds embeddings: {len(bounds_embeddings_df)}")
    print(f"Random embeddings: {len(random_embeddings_df)}")
    
    return bounds_embeddings_df, random_embeddings_df


def finalize_data(
    bounds_embeddings_df: pd.DataFrame,
    random_embeddings_df: pd.DataFrame,
    config: Config
) -> tuple[str, str]:
    """
    Apply LDA and save final data files.
    
    Args:
        bounds_embeddings_df: Bounds embeddings dataframe
        random_embeddings_df: Random embeddings dataframe
        config: Configuration object
        
    Returns:
        Tuple of (bounds_file_path, random_file_path)
    """
    print("\n" + "=" * 80)
    print("Finalizing Data with LDA")
    print("=" * 80)
    
    # Initialize pipeline
    pipeline = DataCreationPipeline(config)
    
    # Run full pipeline
    bounds_path, random_path = pipeline.run_full_pipeline(
        bounds_embeddings_df,
        random_embeddings_df,
        dataset_name=config.TRAIN_DATASET
    )
    
    return bounds_path, random_path


def main():
    """Main data creation pipeline."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create data for species detection pipeline")
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
        "--train-capture",
        type=str,
        help="Override train capture name"
    )
    parser.add_argument(
        "--test-captures",
        nargs="+",
        help="Override test captures list"
    )
    args = parser.parse_args()
    
    # Initialize configuration with optional overrides
    kwargs = {}
    if args.dataset_name:
        kwargs["dataset_name"] = args.dataset_name
    if args.train_capture:
        kwargs["train_capture"] = args.train_capture
    if args.test_captures:
        kwargs["test_captures"] = args.test_captures
    
    config = Config(config_path=args.config, **kwargs)
    
    print("=" * 80)
    print("Species Detection Data Creation Pipeline")
    print("=" * 80)
    print(f"Dataset: {config.TRAIN_DATASET}")
    print(f"Train capture: {config.TRAIN_CAPTURE}")
    print(f"Test captures: {config.TEST_CAPTURES}")
    print(f"Embedding model: {config.DINOv2_MODEL_NAME}")
    
    try:
        # Step 1: Fetch annotations
        train_bounds, test_bounds = fetch_annotations(config)
        
        # Step 2: Process bounds data
        bounds_df = process_bounds_data(train_bounds, test_bounds, config)
        
        # Step 3: Generate embeddings
        bounds_embeddings_df, random_embeddings_df = generate_embeddings(
            bounds_df, config
        )
        
        # Step 5: Finalize data with LDA
        bounds_path, random_path = finalize_data(
            bounds_embeddings_df, random_embeddings_df, config
        )
        
        print("\n" + "=" * 80)
        print("Data Creation Complete!")
        print("=" * 80)
        print(f"Bounds data: {bounds_path}")
        print(f"Random data: {random_path}")
        print(f"Ready for training with: python scripts/train.py")
        
        # Cleanup
        print("\nCleaning up memory...")
        for _ in range(5):
            gc.collect()
        
    except Exception as e:
        print(f"\nError in data creation pipeline: {e}")
        raise


if __name__ == "__main__":
    main()
