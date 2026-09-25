"""
Data preprocessing utilities for image transformations.
"""
import torch
from torchvision.transforms import v2
from typing import Optional


class ImagePreprocessor:
    """Handles image preprocessing transformations."""
    
    def __init__(
        self,
        crop_size: int = 64,
        resize_size: int = 224,
        horizontal_flip_prob: float = 0.5
    ):
        """
        Initialize preprocessor with transformation parameters.
        
        Args:
            crop_size: Size for random crop
            resize_size: Size for resize operation
            horizontal_flip_prob: Probability of horizontal flip
        """
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.horizontal_flip_prob = horizontal_flip_prob
        
        self.transform = v2.Compose([
            v2.RandomCrop(size=(crop_size, crop_size)),
            v2.Resize(size=(resize_size, resize_size), antialias=True),
            v2.RandomHorizontalFlip(p=horizontal_flip_prob),
            v2.ToDtype(torch.float32, scale=True),  # to float32 in [0, 1]
        ])
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply preprocessing transformations to an image.
        
        Args:
            image: Input tensor
            
        Returns:
            Transformed tensor
        """
        return self.transform(image)


class DataSampler:
    """Handles balanced sampling of training/test data."""
    
    @staticmethod
    def prepare_balanced_data(
        data_df,
        target_species: str,
        n_samples: int,
        use_random_negatives: bool = False,
        random_state: int = 42
    ):
        """
        Create balanced dataset with equal positive and negative samples.
        
        Args:
            data_df: Input dataframe
            target_species: Species to consider as positive class
            n_samples: Number of samples per class
            use_random_negatives: If True, negatives are 'random px', else other species
            random_state: Random seed for reproducible sampling
            
        Returns:
            Balanced dataframe
        """
        import pandas as pd
        from sklearn.utils import resample
        
        positive_class = target_species
        
        # Determine negative class: random px if available and requested, otherwise first non-target species
        if use_random_negatives and 'random px' in data_df['name'].values:
            negative_class = 'random px'
        else:
            available_classes = data_df[data_df['name'] != positive_class]['name'].unique()
            if len(available_classes) == 0:
                raise ValueError(f"No negative class found for species '{target_species}'")
            negative_class = available_classes[0]
        
        # Filter positive and negative samples
        positive_data = data_df[data_df['name'] == positive_class]
        negative_data = data_df[data_df['name'] == negative_class]
        
        # Check if we have samples for both classes
        if len(positive_data) == 0:
            raise ValueError(f"No samples found for positive class '{positive_class}'")
        if len(negative_data) == 0:
            raise ValueError(f"No samples found for negative class '{negative_class}'")
        
        # Adjust n_samples if we don't have enough samples
        n_samples_positive = min(n_samples, len(positive_data))
        n_samples_negative = min(n_samples, len(negative_data))
        
        # Ensure we have balanced data by using the minimum of both
        n_samples_balanced = min(n_samples_positive, n_samples_negative)
        
        positive_samples = resample(
            positive_data,
            n_samples=n_samples_balanced,
            random_state=random_state
        )
        
        negative_samples = resample(
            negative_data,
            n_samples=n_samples_balanced,
            random_state=random_state
        )
        
        return pd.concat([positive_samples, negative_samples])
    
    @staticmethod
    def split_train_test(data_df, train_capture: str = '93deg', test_captures: list = ['183deg']):
        """
        Split data into train and test sets based on capture field.
        
        Args:
            data_df: Input dataframe with 'capture' column
            train_capture: Name of capture to use for training
            test_captures: List of captures to use for testing
            
        Returns:
            Tuple of (train_data, test_data)
        """
        train_data = data_df[data_df['capture'] == train_capture]
        test_data = data_df[data_df['capture'].isin(test_captures)]
        
        return train_data, test_data