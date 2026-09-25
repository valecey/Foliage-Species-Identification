"""
Morphological operations for image post-processing.
"""
import numpy as np
from scipy import ndimage
from skimage.morphology import remove_small_holes


class MorphologyProcessor:
    """Applies morphological operations to binary masks."""
    
    @staticmethod
    def open_closing(
        X: np.ndarray,
        structure_size: tuple = (3, 3),
        small_holes_size: int = 10
    ) -> np.ndarray:
        """
        Apply morphological opening and closing operations.
        
        Args:
            X: Input binary array
            structure_size: Size of structuring element
            small_holes_size: Minimum size of holes to keep
            
        Returns:
            Processed binary array
        """
        # Invert (0 becomes True)
        X = X == 0
        X = X.astype('bool')
        
        # Opening: removes small objects
        X = ndimage.binary_opening(X, structure=np.ones(structure_size))
        
        # Closing: fills small holes
        X = ndimage.binary_closing(X, structure=np.ones(structure_size))
        
        # Invert back
        X = ~X
        
        # Remove small holes
        X = remove_small_holes(X, small_holes_size)
        
        X = X.astype('uint8')
        
        return X
    
    @staticmethod
    def remove_small_objects(
        binary_mask: np.ndarray,
        min_size: int = 64
    ) -> np.ndarray:
        """
        Remove small connected components.
        
        Args:
            binary_mask: Input binary mask
            min_size: Minimum size to keep
            
        Returns:
            Filtered binary mask
        """
        from skimage.morphology import remove_small_objects as rso
        
        return rso(binary_mask.astype(bool), min_size=min_size).astype('uint8')
    
    @staticmethod
    def fill_holes(
        binary_mask: np.ndarray,
        area_threshold: int = 64
    ) -> np.ndarray:
        """
        Fill holes in binary mask.
        
        Args:
            binary_mask: Input binary mask
            area_threshold: Maximum hole size to fill
            
        Returns:
            Mask with filled holes
        """
        return remove_small_holes(
            binary_mask.astype(bool),
            area_threshold=area_threshold
        ).astype('uint8')