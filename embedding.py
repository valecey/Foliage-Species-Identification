"""
DINOv2 embedding generation module.
"""
import torch
import numpy as np
import pandas as pd
import os
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from torchvision.transforms import v2
from tqdm.auto import tqdm
from typing import Optional

# Initialize tqdm for pandas
tqdm.pandas()


class EmbeddingGenerator:
    """Generates embeddings using DINOv2 model."""
    
    def __init__(
        self,
        model_name: str = 'facebook/dinov2-base',
        device: str = 'cuda'
    ):
        """
        Initialize embedding generator.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run model on ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device
        
        # Load processor and model
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model = self.model.eval().to(device)
        
        # Create preprocessor ONCE like original
        self.preproc = v2.Compose([
            v2.RandomCrop(size=(64, 64)),
            v2.RandAugment(),
            v2.Resize(size=(224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
        ])
    
    def produce_batch_embeds(
        self,
        crop: torch.Tensor,
        batch_size: int
    ) -> torch.Tensor:
        """
        Generate embeddings for a batch of preprocessed crops.
        Matches original exactly.
        
        Args:
            crop: Input crop tensor of shape (C, H, W)
            batch_size: Number of augmented versions to generate
            
        Returns:
            Embeddings tensor of shape (batch_size, embedding_dim)
        """
        # Generate augmented versions using instance preprocessor
        images = [self.preproc(torch.Tensor(crop)) for i in range(batch_size)]
        
        # Process through model matching original exactly
        with torch.no_grad():
            pixel_values = self.processor(
                images=images,
                return_tensors="pt",
                do_rescale=False
            )['pixel_values'].to(self.device)
            
            embedding = self.model(pixel_values=pixel_values).last_hidden_state[:, 0, :]
        
        return embedding.cpu()
    
    def produce_embeds(
        self,
        crop: torch.Tensor,
        num: int,
        batch_size: int
    ) -> torch.Tensor:
        """
        Generate multiple batches of embeddings.
        Matches original exactly.
        
        Args:
            crop: Input crop tensor
            num: Total number of embeddings to generate
            batch_size: Batch size for processing
            
        Returns:
            Concatenated embeddings tensor
        """
        num_batches = num // batch_size
        embeddings = [
            self.produce_batch_embeds(crop, batch_size)
            for _ in range(num_batches)
        ]
        
        return torch.cat(embeddings)
    
    def to(self, device: str):
        """Move model to specified device."""
        self.device = device
        self.model = self.model.to(device)
        return self
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
        return self
    
    def generate_bounds_embeddings(
        self,
        bounds_df: pd.DataFrame,
        crop_size: int = 64,
        resize_size: int = 224,
        horizontal_flip_prob: float = 0.5,
        num_per_crop: int = 2048,
        batch_size: int = 512
    ) -> pd.DataFrame:
        """
        Generate embeddings for bounding box crops.
        Matches original exactly.
        
        Args:
            bounds_df: DataFrame with crop data
            crop_size: Size for random crop
            resize_size: Size for resize operation
            horizontal_flip_prob: Probability of horizontal flip
            num_per_crop: Number of embeddings per crop
            batch_size: Batch size for processing
            
        Returns:
            DataFrame with embeddings added
        """
        # Generate embeddings for each crop (matching original exactly)
        bounds_df['embeds'] = bounds_df['crop'].progress_apply(
            lambda x: self.produce_embeds(
                np.nan_to_num(x[[0,1,2]]) / np.expand_dims(np.asarray([2**8, 2**8, 2**8]), (-1, -2)),
                num_per_crop,
                batch_size
            )
        )
        
        # Explode embeddings and clean up
        bounds_df['embeds'] = bounds_df['embeds'].apply(lambda x: list(x))
        bounds_df = bounds_df.drop('crop', axis=1)
        bounds_df = bounds_df.explode('embeds')
        
        return bounds_df
    
    def generate_random_patch_embeddings(
        self,
        dataset: np.ndarray,
        n_patches: int = 65536,
        batch_size: int = 512
    ) -> pd.DataFrame:
        """
        Generate embeddings from random patches.
        
        Args:
            dataset: Full image dataset
            n_patches: Number of random patches to generate
            batch_size: Batch size for processing
            
        Returns:
            DataFrame with random patch embeddings
        """
        random_pixels_df = pd.DataFrame()
        random_pixels_df['embeds'] = [
            j for i in tqdm(range(n_patches // 2048)) 
            for j in self.produce_embeds(
                np.nan_to_num(dataset[[0,1,2]]) / np.expand_dims(np.asarray([2**8, 2**8, 2**8]), (-1, -2)),
                2048 // 4,
                batch_size
            )
        ]
        
        return random_pixels_df
    
    @staticmethod
    def save_crops_to_disk(
        bounds_df: pd.DataFrame,
        output_dir: str = 'crops'
    ):
        """
        Save crops as individual image files.
        
        Args:
            bounds_df: DataFrame with crop data
            output_dir: Directory to save crops
        """
        # Create directories for each unique label
        for label in bounds_df['name'].unique():
            os.makedirs(f'{output_dir}/{label}', exist_ok=True)
        
        # Save each crop as an image file
        for idx, row in bounds_df.iterrows():
            label = row['name']
            filename = f"{row['ds']}_{row['capture']}_{idx}.png"
            save_dir = f'{output_dir}/{label}'
            
            # Convert crop to RGB image (first 3 channels normalized to 0-255)
            img_array = (row['crop'][:3].transpose(1,2,0) * 255).astype('uint8')
            img = Image.fromarray(img_array)
            img.save(os.path.join(save_dir, filename))