"""
Label Studio format conversion utilities.
"""
import json
import numpy as np
from skimage import measure
from typing import List, Dict, Any


class LabelStudioConverter:
    """Converts predictions to Label Studio format."""
    
    @staticmethod
    def create_rectangle_annotation(
        bbox: tuple,
        image_shape: tuple,
        species_name: str,
        region_idx: int,
        species_idx: int,
        capture_idx: int,
        vote_count: int
    ) -> Dict[str, Any]:
        """
        Create a single rectangle annotation in Label Studio format.
        
        Args:
            bbox: Bounding box (min_row, min_col, max_row, max_col)
            image_shape: Shape of image (height, width)
            species_name: Name of species
            region_idx: Index of region
            species_idx: Index of species
            capture_idx: Index of capture
            vote_count: Number of pixels in region
            
        Returns:
            Annotation dictionary
        """
        min_row, min_col, max_row, max_col = bbox
        height, width = image_shape
        
        # Convert to normalized coordinates (0-100)
        x_min = min_col / width * 100
        y_min = min_row / height * 100
        x_max = max_col / width * 100
        y_max = max_row / height * 100
        width_pct = x_max - x_min
        height_pct = y_max - y_min
        
        result = {
            "id": f"{species_idx}_{capture_idx}_{region_idx}",
            "type": "rectanglelabels",
            "value": {
                "x": x_min,
                "y": y_min,
                "width": width_pct,
                "height": height_pct,
                "rotation": 0,
                "rectanglelabels": [species_name],
                "vote_count": vote_count
            },
            "origin": "manual",
            "to_name": "image",
            "from_name": "label",
            "image_rotation": 0,
            "original_width": width,
            "original_height": height
        }
        
        return result
    
    @staticmethod
    def mask_to_annotations(
        mask: np.ndarray,
        species_name: str,
        species_idx: int,
        capture_idx: int
    ) -> List[Dict[str, Any]]:
        """
        Convert binary mask to list of rectangle annotations.
        
        Args:
            mask: Binary mask array
            species_name: Name of species
            species_idx: Index of species
            capture_idx: Index of capture
            
        Returns:
            List of annotation dictionaries
        """
        # Handle different dimensions in mask
        if len(mask.shape) > 2:
            mask_2d = np.sum(mask, axis=-1)
            mask_2d = (mask_2d > 0).astype('uint8')
        else:
            mask_2d = mask
        
        # Find contiguous regions
        labeled_mask, num_labels = measure.label(
            mask_2d, connectivity=2, return_num=True
        )
        region_props = measure.regionprops(labeled_mask)
        
        # Get image dimensions
        height, width = mask_2d.shape
        
        # Create annotations for each region
        annotations = []
        for region_idx, region in enumerate(region_props):
            bbox = region.bbox
            
            # Handle different bbox formats
            if len(bbox) == 4:
                min_row, min_col, max_row, max_col = bbox
            elif len(bbox) == 6:  # For 3D images
                min_row, min_col, _, max_row, max_col, _ = bbox
            else:
                print(f"Warning: Unexpected bbox format: {bbox}")
                continue
            
            vote_count = region.area
            
            annotation = LabelStudioConverter.create_rectangle_annotation(
                bbox=(min_row, min_col, max_row, max_col),
                image_shape=(height, width),
                species_name=species_name,
                region_idx=region_idx,
                species_idx=species_idx,
                capture_idx=capture_idx,
                vote_count=vote_count
            )
            
            annotations.append(annotation)
            
            print(f"region index:{region_idx} and vote count: {vote_count} - "
                  f"image size: {height}x{width}")
        
        return annotations
    
    @staticmethod
    def create_image_annotation(
        species_idx: int,
        capture_idx: int,
        annotations: List[Dict[str, Any]],
        file_upload: str = ""
    ) -> Dict[str, Any]:
        """
        Create complete image annotation with all regions.
        
        Args:
            species_idx: Index of species
            capture_idx: Index of capture
            annotations: List of annotation results
            file_upload: File upload path (optional)
            
        Returns:
            Image annotation dictionary
        """
        return {
            "id": species_idx * 100 + capture_idx,
            "file_upload": file_upload,
            "annotations": [{
                "id": species_idx * 100 + capture_idx,
                "result": annotations,
                "was_cancelled": False,
                "ground_truth": False,
                "created_at": "2025-04-16T00:00:00.000000Z",
                "updated_at": "2025-04-16T00:00:00.000000Z",
                "lead_time": 0,
                "prediction": {},
                "result_count": len(annotations)
            }]
        }
    
    @staticmethod
    def save_to_json(
        data: List[Dict[str, Any]],
        filepath: str,
        indent: int = 2
    ):
        """
        Save Label Studio data to JSON file.
        
        Args:
            data: List of image annotations
            filepath: Output file path
            indent: JSON indentation level
        """
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=indent)
        
        print(f"Created {len(data)} image annotations in Label Studio format")
        print(f"Saved to {filepath}")