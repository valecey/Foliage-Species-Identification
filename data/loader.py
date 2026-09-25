"""
Data loading utilities for geospatial imagery.
Handles both TIF and PNG formats with consistent interface.
"""
from pathlib import Path
from typing import List, Union
import numpy as np
import pandas as pd
from PIL import Image
import rasterio as rio


# Disable PIL image size limit for large images
Image.MAX_IMAGE_PIXELS = None


class DataLoader:
    """Handles loading of geospatial imagery datasets."""
    
    @staticmethod
    def load_dataset(ds_path: Path, band_filenames: List[str]) -> np.ndarray:
        """
        Load a dataset from the given path with specified band filenames.
        
        Args:
            ds_path: Path to dataset directory
            band_filenames: List of filenames to load
            
        Returns:
            Concatenated array of shape (channels, height, width)
        """
        filenames = [ds_path / fname for fname in band_filenames]
        arrs = []
        
        for filepath in filenames:
            arr = DataLoader._load_single_file(filepath)
            arrs.append(arr)
        
        # Handle RGB channel selection for TIF files
        if not str(filenames[0]).lower().endswith('.png'):
            arrs[0] = arrs[0][:3]
        
        # Concatenate all arrays along channel dimension
        return np.concatenate(arrs)
    
    @staticmethod
    def _load_single_file(filepath: Path) -> np.ndarray:
        """
        Load a single image file (PNG or TIF).
        
        Args:
            filepath: Path to image file
            
        Returns:
            Array of shape (channels, height, width)
        """
        if str(filepath).lower().endswith('.png'):
            return DataLoader._load_png(filepath)
        else:
            return DataLoader._load_tif(filepath)
    
    @staticmethod
    def _load_png(filepath: Path) -> np.ndarray:
        """
        Load PNG file and convert to CHW format.
        
        Args:
            filepath: Path to PNG file
            
        Returns:
            Array of shape (channels, height, width)
        """
        arr = np.array(Image.open(filepath))
        
        if arr.ndim == 2:  # Grayscale
            arr = arr[None, ...]
        elif arr.ndim == 3:  # Color (HWC -> CHW)
            arr = arr.transpose(2, 0, 1)
        
        return arr
    
    @staticmethod
    def _load_tif(filepath: Path) -> np.ndarray:
        """
        Load TIF file using rasterio.
        
        Args:
            filepath: Path to TIF file
            
        Returns:
            Array of shape (channels, height, width)
        """
        return rio.open(filepath).read()


class DataFrameLoader:
    """Handles loading of preprocessed dataframes."""
    
    @staticmethod
    def load_embeddings(bounds_path: str, random_path: str) -> tuple:
        """
        Load bounds and random pixel embeddings dataframes.
        
        Args:
            bounds_path: Path to bounds embeddings pickle
            random_path: Path to random embeddings pickle
            
        Returns:
            Tuple of (bounds_df, random_px_df)
        """
        import pandas as pd
        
        bounds_df = pd.read_pickle(bounds_path)
        random_px_df = pd.read_pickle(random_path)
        
        # Fill NaN values
        bounds_df = bounds_df.fillna(0)
        random_px_df = random_px_df.fillna(0)
        
        return bounds_df, random_px_df
    
    @staticmethod
    def filter_species(bounds_df, species_list: List[str]):
        """
        Filter dataframe to include only specified species and remove unknowns.
        
        Args:
            bounds_df: Bounds dataframe
            species_list: List of species to keep
            
        Returns:
            Filtered dataframe with 'Other' category for non-matching species
        """
        import pandas as pd
        
        # Remove unknown species
        bounds_df = bounds_df[bounds_df['name'] != '_Unknown']
        
        # Convert non-target species to 'Other' using .loc to avoid SettingWithCopyWarning
        bounds_df = bounds_df.copy()  # Create a copy to ensure we're not modifying a view
        bounds_df.loc[:, 'name'] = bounds_df['name'].apply(
            lambda x: 'Other' if x not in species_list else x
        )
        
        return bounds_df