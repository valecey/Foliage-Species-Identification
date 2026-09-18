"""
Dimensionality reduction modules for embedding processing.
"""
import torch
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from typing import Tuple, Optional


class LDAReducer:
    """Handles Linear Discriminant Analysis for dimensionality reduction."""
    
    def __init__(self):
        """Initialize LDA reducer."""
        self.lda = None
        self.label_encoder = LabelEncoder()
    
    def fit_lda(
        self,
        bounds_df: pd.DataFrame,
        feature_cols: list,
        label_col: str = 'name',
        dataset_filter: Optional[str] = None,
        capture_filter: Optional[str] = None
    ) -> LinearDiscriminantAnalysis:
        """
        Fit LDA on bounds data.
        
        Args:
            bounds_df: DataFrame with embeddings and labels
            feature_cols: Column names for embedding features
            label_col: Column name for labels
            dataset_filter: Filter by dataset name
            capture_filter: Filter by capture name
            
        Returns:
            Fitted LDA model
        """
        # Filter data if specified
        train_data = bounds_df.copy()
        if dataset_filter:
            train_data = train_data[train_data['ds'] == dataset_filter]
        if capture_filter:
            train_data = train_data[train_data['capture'] == capture_filter]
        
        # Fit LDA
        self.lda = LinearDiscriminantAnalysis()
        self.lda.fit(train_data[feature_cols], train_data[label_col])
        
        return self.lda
    
    def transform_embeddings(
        self,
        df: pd.DataFrame,
        feature_cols: list,
        keep_original_cols: bool = True
    ) -> pd.DataFrame:
        """
        Apply LDA transformation to embeddings.
        
        Args:
            df: DataFrame with embeddings
            feature_cols: Column names for embedding features
            keep_original_cols: Whether to keep original embedding columns
            
        Returns:
            DataFrame with LDA-transformed embeddings
        """
        if self.lda is None:
            raise ValueError("LDA must be fitted before transformation")
        
        # Transform embeddings
        transformed_features = self.lda.transform(df[feature_cols])
        
        # Create result dataframe
        if keep_original_cols:
            result_df = df.drop(feature_cols, axis=1).reset_index(drop=True)
        else:
            result_df = pd.DataFrame()
        
        # Add transformed features
        transformed_df = pd.DataFrame(transformed_features).reset_index(drop=True)
        result_df = pd.concat([result_df, transformed_df], axis=1)
        
        return result_df
    
    def save_lda_model(self, path: str):
        """Save LDA model to file."""
        if self.lda is None:
            raise ValueError("No LDA model to save")
        torch.save(self.lda, path)
        print(f"LDA model saved to {path}")
    
    def load_lda_model(self, path: str):
        """Load LDA model from file."""
        self.lda = torch.load(path)
        print(f"LDA model loaded from {path}")
        return self.lda


class EmbeddingProcessor:
    """Processes embeddings for training pipeline."""
    
    @staticmethod
    def prepare_bounds_dataframe(
        bounds_df: pd.DataFrame,
        embedding_col: str = 'embeds'
    ) -> pd.DataFrame:
        """
        Prepare bounds dataframe by converting embeddings to columns.
        
        Args:
            bounds_df: DataFrame with embeddings in list/tensor format
            embedding_col: Column name containing embeddings
            
        Returns:
            DataFrame with embeddings as separate columns
        """
        # Convert embeddings to separate columns
        embeddings_tensor = torch.stack(list(bounds_df[embedding_col]))
        embeddings_df = pd.DataFrame(embeddings_tensor.numpy())
        
        # Combine with original data (excluding embedding column)
        result_df = pd.concat([
            bounds_df.reset_index(drop=True).drop(embedding_col, axis=1),
            embeddings_df
        ], axis=1)
        
        return result_df
    
    @staticmethod
    def finalize_dataframes(
        bounds_df: pd.DataFrame,
        random_px_df: pd.DataFrame,
        n_features: int = 768
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Finalize dataframes by selecting correct columns.
        
        Args:
            bounds_df: Bounds dataframe
            random_px_df: Random pixels dataframe
            n_features: Number of embedding features
            
        Returns:
            Tuple of (bounds_df, random_px_df) with correct columns
        """
        # Select first n_features columns for embeddings
        bounds_final = bounds_df.iloc[:, :8 + n_features].copy()
        random_final = random_px_df.iloc[:, :3 + n_features].copy()
        
        return bounds_final, random_final
    
    @staticmethod
    def save_processed_data(
        bounds_df: pd.DataFrame,
        random_px_df: pd.DataFrame,
        dataset_name: str,
        use_lda: bool = True
    ):
        """
        Save processed dataframes to pickle files.
        
        Args:
            bounds_df: Processed bounds dataframe
            random_px_df: Processed random pixels dataframe
            dataset_name: Name of dataset
            use_lda: Whether data includes LDA transformation
        """
        suffix = "_lda_v2_dinov2_rgb" if use_lda else "_dinov2_rgb"
        
        bounds_path = f'{dataset_name}_bounds_embeddings{suffix}.pickle'
        random_path = f'{dataset_name}_random_embeddings{suffix}.pickle'
        
        # Save with compatible protocol
        bounds_df.to_pickle(bounds_path, protocol=4)
        random_px_df.to_pickle(random_path, protocol=4)
        
        print(f"Saved bounds data to {bounds_path}")
        print(f"Saved random pixels data to {random_path}")


class DataCreationPipeline:
    """Complete pipeline for creating training data from raw images."""
    
    def __init__(self, config):
        """Initialize pipeline with configuration."""
        self.config = config
        self.lda_reducer = LDAReducer()
        self.embedding_processor = EmbeddingProcessor()
    
    def run_full_pipeline(
        self,
        bounds_df: pd.DataFrame,
        random_px_df: pd.DataFrame,
        dataset_name: str = 'lp3'
    ) -> Tuple[str, str]:
        """
        Run the complete data creation pipeline.
        
        Args:
            bounds_df: DataFrame with bounds embeddings
            random_px_df: DataFrame with random pixel embeddings
            dataset_name: Name of the dataset
            
        Returns:
            Tuple of (bounds_file_path, random_file_path)
        """
        print("Starting full data creation pipeline...")
        
        # Step 1: Prepare dataframes
        print("Preparing dataframes...")
        bounds_df = self.embedding_processor.prepare_bounds_dataframe(bounds_df)
        random_px_df = self.embedding_processor.prepare_bounds_dataframe(random_px_df)
        
        # Step 2: Finalize dataframes
        print("Finalizing dataframes...")
        bounds_df, random_px_df = self.embedding_processor.finalize_dataframes(
            bounds_df, random_px_df
        )
        
        # Step 3: Fit LDA on training data
        print("Fitting LDA...")
        feature_cols = list(range(768))  # DinoV2 embedding dimension
        self.lda_reducer.fit_lda(
            bounds_df, 
            feature_cols, 
            dataset_filter=dataset_name
        )
        
        # Step 4: Save LDA model
        lda_model_path = f'lda_v2_{dataset_name}_dinov2_rgb.pickle'
        self.lda_reducer.save_lda_model(lda_model_path)
        
        # Step 5: Transform embeddings with LDA
        print("Transforming embeddings with LDA...")
        bounds_df_lp3 = bounds_df[bounds_df['ds'] == dataset_name].copy()
        bounds_df_lp3 = self.lda_reducer.transform_embeddings(bounds_df_lp3, feature_cols)
        
        random_px_df_lp3 = random_px_df[random_px_df['ds'] == dataset_name].copy()
        random_px_df_lp3 = self.lda_reducer.transform_embeddings(random_px_df_lp3, feature_cols)
        
        # Step 6: Save processed data
        print("Saving processed data...")
        self.embedding_processor.save_processed_data(
            bounds_df_lp3, random_px_df_lp3, dataset_name, use_lda=True
        )
        
        bounds_path = f'{dataset_name}_bounds_embeddings_lda_v2_dinov2_rgb.pickle'
        random_path = f'{dataset_name}_random_embeddings_lda_v2_dinov2_rgb.pickle'
        
        print("Pipeline completed successfully!")
        return bounds_path, random_path
