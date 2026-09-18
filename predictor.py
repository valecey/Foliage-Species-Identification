"""
Prediction and inference module for trained models.
"""
import torch
import numpy as np
from tqdm.auto import tqdm
from einops import rearrange
from typing import Tuple, Optional


class ModelPredictor:
    """Handles batch prediction on large images."""
    
    def __init__(
        self,
        embedding_generator,
        lda_transformer,
        patch_size: int = 64,
        binary_mode: bool = False
    ):
        """
        Initialize predictor.
        
        Args:
            embedding_generator: EmbeddingGenerator instance
            lda_transformer: Trained LDA transformer
            patch_size: Size of patches to extract
            binary_mode: Whether to use binary mode (extract only first LDA component)
        """
        self.embedding_generator = embedding_generator
        self.lda_transformer = lda_transformer
        self.patch_size = patch_size
        self.binary_mode = binary_mode
    
    def batched_pred(
        self,
        clf_tuple: Tuple,
        image: np.ndarray,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Predict on an entire image using patch-based approach.
        
        Args:
            clf_tuple: Tuple of (scaler, label_encoder, classifier)
            image: Input image of shape (C, H, W)
            show_progress: Whether to show progress bar
            
        Returns:
            Prediction array of shape (H_patches, W_patches, ...)
        """
        # Crop to multiple of patch_size
        h_crop = image.shape[1] - (image.shape[1] % self.patch_size)
        w_crop = image.shape[2] - (image.shape[2] % self.patch_size)
        image = image[:, :h_crop, :w_crop]
        
        # Extract patches
        batched = [
            patch for patch in rearrange(
                image,
                'c (h h1) (w w1) -> (h w) c h1 w1',
                h1=self.patch_size,
                w1=self.patch_size
            )
        ]
        
        # Get classifier
        classifier = clf_tuple[2]
        
        # Predict on each patch
        predictions = []
        iterator = tqdm(batched) if show_progress else batched
        
        with torch.no_grad():
            for patch in iterator:
                # Normalize patch to match original create_maca.py
                patch_normalized = np.nan_to_num(patch[[0,1,2]]) / np.expand_dims(np.asarray([2**8, 2**8, 2**8]), (-1, -2))
                
                # Generate embedding
                embedding = self.embedding_generator.produce_batch_embeds(
                    patch_normalized,
                    batch_size=1
                ).numpy()
                
                # Transform with LDA
                transformed = self.lda_transformer.transform(embedding)
                
                # Handle binary mode - extract only first component if needed
                if self.binary_mode and transformed.shape[1] > 1:
                    transformed = transformed[:, :1]
                
                # Predict
                pred = classifier.predict(transformed)
                predictions.append(pred)
        
        # Reshape predictions
        num_h_patches = image.shape[1] // self.patch_size
        num_w_patches = image.shape[2] // self.patch_size
        
        predictions = np.concatenate(predictions).reshape(
            (num_h_patches, num_w_patches, -1)
        )
        
        return predictions
    
    def predict_image(
        self,
        image: np.ndarray,
        clf_tuple: Tuple,
        binary_threshold: int = 0
    ) -> np.ndarray:
        """
        Predict and return binary mask.
        
        Args:
            image: Input image
            clf_tuple: Classifier tuple
            binary_threshold: Threshold for binarization
            
        Returns:
            Binary mask
        """
        # Handle NaN values
        image = np.nan_to_num(image)
        
        # Get predictions
        img_pred = self.batched_pred(clf_tuple, image)
        
        # Binarize
        img_pred = (img_pred == binary_threshold).astype('uint8')
        
        return img_pred
    
    def predict_multiple_species(
        self,
        image: np.ndarray,
        species_models: dict,
        binary_threshold: int = 0
    ) -> dict:
        """
        Predict for multiple species.
        
        Args:
            image: Input image
            species_models: Dictionary mapping species names to model tuples
            binary_threshold: Threshold for binarization
            
        Returns:
            Dictionary mapping species names to prediction masks
        """
        predictions = {}
        
        for species_name, clf_tuple in species_models.items():
            print(f"Predicting for {species_name}...")
            predictions[species_name] = self.predict_image(
                image, clf_tuple, binary_threshold
            )
        
        return predictions