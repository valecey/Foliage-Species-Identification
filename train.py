"""
Training script for Macaranga detection models.
Refactored from train_macca.py with modular structure.
"""
import argparse
import gc
import pickle
import torch
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.config import Config
from data.loader import DataLoader, DataFrameLoader
from data.preprocessor import ImagePreprocessor, DataSampler
from models.svm_trainer import SVMTrainer


def prepare_training_data(bounds_df, random_px_df, species_name, use_biased_svm=False):
    """
    Prepare training data for a specific species.
    
    Args:
        bounds_df: Bounds dataframe
        random_px_df: Random pixels dataframe
        species_name: Target species name
        use_biased_svm: Whether to use biased SVM approach
        
    Returns:
        Combined dataframe ready for training
    """
    # Separate target species from others
    target_df = bounds_df[bounds_df['name'] == species_name].copy()
    not_target_df = bounds_df[bounds_df['name'] != species_name].copy()
    
    print(f"Processing species: {species_name}")
    
    # Set up negative class
    if not use_biased_svm:
        not_target_df['real_name'] = not_target_df['name']
        not_target_df['name'] = 'Not Selected'
        data_ds = pd.concat([target_df, not_target_df])
    else:
        random_px_df['name'] = 'random px'
        data_ds = pd.concat([target_df, random_px_df])
    
    return data_ds


def train_single_species(
    data_ds,
    species_name,
    config: Config
):
    """
    Train model for a single species.
    
    Args:
        data_ds: Combined training data
        species_name: Target species name
        config: Configuration object
        
    Returns:
        Tuple of (scaler, label_encoder, trained_model)
    """
    # Split train and test
    train_data, test_data = DataSampler.split_train_test(
        data_ds, 
        train_capture=config.TRAIN_CAPTURE,
        test_captures=config.TEST_CAPTURES
    )
    
    # If no test data for this species, use training data for testing (fallback)
    if len(test_data[test_data['name'] == species_name]) == 0:
        print(f"Warning: No test data found for species '{species_name}'. Using training data for evaluation.")
        test_data = train_data.copy()
    
    # Balance datasets
    train_data = DataSampler.prepare_balanced_data(
        train_data,
        target_species=species_name,
        n_samples=config.N_SAMPLES_TRAIN,
        use_random_negatives=config.USE_BIASED_SVM,
        random_state=config.RANDOM_STATE
    )
    
    test_data = DataSampler.prepare_balanced_data(
        test_data,
        target_species=species_name,
        n_samples=config.N_SAMPLES_TEST,
        use_random_negatives=config.USE_BIASED_SVM,
        random_state=config.RANDOM_STATE
    )
    
    # Extract features and labels
    embedding_cols = [col for col in train_data.columns if isinstance(col, int)]
    train_X = train_data[embedding_cols].to_numpy()
    train_y = train_data['name'].to_numpy()
    test_X = test_data[embedding_cols].to_numpy()
    test_y = test_data['name'].to_numpy()
    
    print(f"Training data shape: {train_X.shape}, Test data shape: {test_X.shape}")
    
    # Initialize trainer
    if config.USE_BIASED_SVM:
        gamma_values = config.GAMMA_VALUES_BIASED
        c_values = config.C_VALUES_BIASED
        w_c_values = config.W_C_VALUES
    else:
        gamma_values = config.GAMMA_VALUES_STANDARD
        c_values = config.C_VALUES_STANDARD
        w_c_values = None
    
    trainer = SVMTrainer(
        gamma_values=gamma_values,
        c_values=c_values,
        use_biased_svm=config.USE_BIASED_SVM,
        w_c_values=w_c_values,
        sensitivity_target=config.SENSITIVITY_TARGET,
        n_splits=config.N_SPLITS_CV,
        verbosity=config.VERBOSITY,
        n_jobs=config.N_JOBS,
        random_state=config.RANDOM_STATE
    )
    
    # Determine negative class name
    negative_class = 'random px' if config.USE_BIASED_SVM else 'Not Selected'
    
    # Train model
    sc, le, clf = trainer.train(
        train_X=train_X,
        train_y=train_y,
        test_X=test_X,
        test_y=test_y,
        species_name=species_name,
        negative_class_name=negative_class
    )
    
    return sc, le, clf


def main():
    """Main training pipeline."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train species detection models")
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
        "--species-list",
        nargs="+",
        help="Override species list"
    )
    args = parser.parse_args()
    
    # Initialize configuration with optional overrides
    kwargs = {}
    if args.dataset_name:
        kwargs["dataset_name"] = args.dataset_name
    if args.species_list:
        kwargs["species_list"] = args.species_list
    
    config = Config(config_path=args.config, **kwargs)
    
    print("=" * 80)
    print("Macaranga Detection Model Training")
    print("=" * 80)
    
    # Load datasets
    print("\nLoading datasets...")
    lp3_first_run_ds = DataLoader.load_dataset(
        config.LP3_FIRST_RUN,
        config.BAND_FILENAMES_PNG
    )
    lp3_m3m_ds = DataLoader.load_dataset(
        config.LP3_M3M,
        config.BAND_FILENAMES_PNG
    )
    print(f"Loaded training dataset: {lp3_first_run_ds.shape}")
    print(f"Loaded test dataset: {lp3_m3m_ds.shape}")
    
    # Load embeddings
    print("\nLoading embeddings...")
    bounds_df, random_px_df = DataFrameLoader.load_embeddings(
        config.BOUNDS_DF_PATH,
        config.RANDOM_PX_DF_PATH
    )
    print(f"Bounds DF shape: {bounds_df.shape}")
    print(f"Random pixels DF shape: {random_px_df.shape}")
    
    # Filter species
    bounds_df = DataFrameLoader.filter_species(bounds_df, config.SPECIES_LIST)
    # Only train on target species, not 'Other'
    species_list = [species for species in config.SPECIES_LIST if species in bounds_df['name'].values]
    print(f"\nSpecies to train: {species_list}")
    
    # Create model directory
    model_dir_path = Path(config.SVM_MODEL_DIR)
    model_dir_path.mkdir(parents=True, exist_ok=True)
    print(f"\nModel output directory: {config.SVM_MODEL_DIR}")
    
    # Train models for each species
    print("\n" + "=" * 80)
    print("Training Models")
    print("=" * 80)
    
    for species in tqdm(species_list, desc="Training species"):
        print(f"\n{'='*60}")
        print(f"Training model for: {species}")
        print(f"{'='*60}")
        
        # Prepare data
        data_ds = prepare_training_data(
            bounds_df,
            random_px_df,
            species,
            use_biased_svm=config.USE_BIASED_SVM
        )
        
        # Train model
        sc, le, clf = train_single_species(data_ds, species, config)
        
        # Save model
        model_path = Path(config.SVM_MODEL_DIR) / species
        with open(model_path, 'wb') as f:
            pickle.dump((sc, le, clf), f)
        
        print(f"✓ Model saved to: {model_path}")
        
        # Garbage collection
        for _ in range(3):
            gc.collect()
    
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)
    print(f"Models saved in: {config.SVM_MODEL_DIR}")


if __name__ == "__main__":
    main()