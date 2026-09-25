"""
Visualization utilities for predictions and results.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image, ImageDraw
from skimage import measure
from typing import List, Dict, Tuple, Optional


class Visualizer:
    """Handles visualization of predictions and masks."""
    
    @staticmethod
    def create_figure(
        num_species: int,
        num_captures: int = 2,
        figsize_per_row: Tuple[float, float] = (10, 4.8),
        wspace: float = 0.1
    ) -> Tuple[plt.Figure, np.ndarray]:
        """
        Create figure for multi-species visualization.
        
        Args:
            num_species: Number of species to plot
            num_captures: Number of capture types per species
            figsize_per_row: Figure size per row
            wspace: Horizontal spacing between subplots
            
        Returns:
            Tuple of (figure, axes array)
        """
        fig, ax = plt.subplots(
            num_species,
            num_captures,
            figsize=(figsize_per_row[0], figsize_per_row[1] * num_species)
        )
        
        # Force ax to be 2D if single species
        if num_species == 1:
            ax = np.array([ax]).reshape(1, -1)
        
        # Reduce spacing between subplots
        plt.subplots_adjust(wspace=wspace)
        
        return fig, ax
    
    @staticmethod
    def plot_image_with_predictions(
        ax: plt.Axes,
        image: np.ndarray,
        title: str = ""
    ):
        """
        Plot RGB image on axes.
        
        Args:
            ax: Matplotlib axes
            image: Image array of shape (H, W, 3)
            title: Title for the plot
        """
        if image.dtype == np.uint8:
            ax.imshow(image)
        else:
            ax.imshow(image / 255)
        
        if title:
            ax.set_title(title)
    
    @staticmethod
    def plot_binary_mask(
        ax: plt.Axes,
        mask: np.ndarray,
        add_bounding_boxes: bool = False,
        box_color: str = 'red',
        box_linewidth: float = 1
    ) -> List[Dict]:
        """
        Plot binary mask with optional bounding boxes.
        
        Args:
            ax: Matplotlib axes
            mask: Binary mask array
            add_bounding_boxes: Whether to add bounding boxes
            box_color: Color of bounding boxes
            box_linewidth: Line width for boxes
            
        Returns:
            List of region properties dictionaries
        """
        # Handle different dimensions in mask
        if len(mask.shape) > 2:
            # Flatten by summing across channels
            mask_2d = np.sum(mask, axis=-1)
            mask_2d = (mask_2d > 0).astype('uint8')
        else:
            mask_2d = mask
        
        # Display mask as RGB
        mask_rgb = np.repeat(mask_2d[:, :, np.newaxis], 3, axis=2) * 255
        mask_rgb = mask_rgb.astype('uint8')
        
        ax.imshow(mask_rgb / 255)
        
        # Add bounding boxes if requested
        regions = []
        if add_bounding_boxes:
            labeled_mask, num_labels = measure.label(
                mask_2d, connectivity=2, return_num=True
            )
            region_props = measure.regionprops(labeled_mask)
            
            for region in region_props:
                bbox = region.bbox
                
                # Handle different bbox formats
                if len(bbox) == 4:
                    min_row, min_col, max_row, max_col = bbox
                elif len(bbox) == 6:  # For 3D images
                    min_row, min_col, _, max_row, max_col, _ = bbox
                else:
                    print(f"Warning: Unexpected bbox format: {bbox}")
                    continue
                
                # Draw rectangle
                rect = Rectangle(
                    (min_col, min_row),
                    max_col - min_col,
                    max_row - min_row,
                    fill=False,
                    edgecolor=box_color,
                    linewidth=box_linewidth
                )
                ax.add_patch(rect)
                
                # Store region info
                regions.append({
                    'bbox': (min_row, min_col, max_row, max_col),
                    'area': region.area,
                    'centroid': region.centroid
                })
        
        return regions
    
    @staticmethod
    def save_figure(
        fig: plt.Figure,
        filepath: str,
        dpi: int = 300,
        bbox_inches: str = 'tight'
    ):
        """
        Save figure to file.
        
        Args:
            fig: Matplotlib figure
            filepath: Output file path
            dpi: Resolution in dots per inch
            bbox_inches: Bounding box setting
        """
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
        print(f"Saved figure to {filepath}")
    
    @staticmethod
    def prepare_rgb_image(
        image_array: np.ndarray,
        channels_first: bool = True
    ) -> np.ndarray:
        """
        Prepare image for visualization (convert to HWC uint8).
        
        Args:
            image_array: Input image array
            channels_first: Whether input is in CHW format
            
        Returns:
            RGB image in HWC format as uint8
        """
        if channels_first:
            # CHW -> HWC
            rgb = image_array[:3].transpose(1, 2, 0)
        else:
            rgb = image_array[:, :, :3]
        
        # Ensure uint8
        if rgb.dtype != np.uint8:
            if rgb.max() <= 1.0:
                rgb = (rgb * 255).astype('uint8')
            else:
                rgb = rgb.astype('uint8')
        
        return rgb