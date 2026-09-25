"""
Label Studio annotation fetching and processing utilities.
Extracts bounding box annotations from Label Studio API.
"""
import json
import pandas as pd
import requests
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


class AnnotationFetcher:
    """Handles fetching and processing annotations from Label Studio."""
    
    def __init__(self, config):
        """
        Initialize annotation fetcher.
        
        Args:
            config: Configuration object with Label Studio settings
        """
        self.config = config
    
    def fetch_annotation(self, task_id: int, output_file: str) -> dict:
        """
        Fetch annotation from Label Studio API.
        
        Args:
            task_id: Label Studio task ID
            output_file: File to save annotation JSON
            
        Returns:
            Annotation data dictionary
        """
        url = f"{self.config.LABEL_STUDIO_URL}/api/tasks/{task_id}/annotations"
        headers = {"Authorization": f"Token {self.config.LABEL_STUDIO_API_KEY}"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        data = response.json()
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        return data
    
    @staticmethod
    def polygon_to_bbox(points: List[List[float]]) -> Tuple[float, float, float, float]:
        """
        Convert polygon points to bounding box.
        
        Args:
            points: List of [x, y] coordinate pairs
            
        Returns:
            Tuple of (x_min, y_min, x_max, y_max)
        """
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        return min(xs), min(ys), max(xs), max(ys)
    
    def process_annotations_to_csv(
        self,
        annotation_data,
        output_csv_path: str
    ) -> pd.DataFrame:
        """
        Process annotation data to CSV format.
        Handles both list and dict responses from API.
        
        Args:
            annotation_data: Annotation data from API (list or dict)
            output_csv_path: Path to save CSV file
            
        Returns:
            DataFrame with bounding box information
        """
        rows = []
        
        # Handle both list and dict responses
        annotations = annotation_data if isinstance(annotation_data, list) else annotation_data.get('annotations', [])
        
        for ann in annotations:
            for result in ann.get('result', []):
                if result.get('type') == 'polygonlabels':
                    label = result['value']['polygonlabels'][0]
                    points = result['value']['points']
                    width = result['original_width']
                    height = result['original_height']
                    
                    # Convert percent to pixel if needed
                    if max([max(xs) for xs in points]) <= 100:
                        points_px = [[x/100*width, y/100*height] for x, y in points]
                    else:
                        points_px = points
                    
                    x0, y0, x1, y1 = self.polygon_to_bbox(points_px)
                    rows.append({
                        'name': label,
                        'y0': int(round(y0)),
                        'y1': int(round(y1)),
                        'x0': int(round(x0)),
                        'x1': int(round(x1)),
                    })
        
        df = pd.DataFrame(rows)
        df.to_csv(output_csv_path, index=False)
        print(f"Saved {len(df)} bounding boxes to {output_csv_path}")
        return df


class BoundsProcessor:
    """Processes bounding box data for training."""
    
    @staticmethod
    def add_metadata(
        bounds_df: pd.DataFrame, 
        dataset_name: str, 
        capture_name: str
    ) -> pd.DataFrame:
        """
        Add dataset and capture metadata to bounds dataframe.
        
        Args:
            bounds_df: Input bounds dataframe
            dataset_name: Dataset name (e.g., 'lp3')
            capture_name: Capture name (e.g., 'first_run')
            
        Returns:
            DataFrame with added metadata columns
        """
        df = bounds_df.copy()
        df['ds'] = dataset_name
        df['capture'] = capture_name
        return df
    
    @staticmethod
    def produce_crop(dataset: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        """
        Extract crop from dataset array.
        
        Args:
            dataset: Full image dataset array (C, H, W)
            y0, y1: Y coordinate bounds
            x0, x1: X coordinate bounds
            
        Returns:
            Cropped array
        """
        return np.nan_to_num(dataset[:, y0:y1, x0:x1])
    
    def add_crops_to_dataframe(
        self, 
        bounds_df: pd.DataFrame, 
        dataset: np.ndarray
    ) -> pd.DataFrame:
        """
        Add crop data to bounds dataframe.
        
        Args:
            bounds_df: DataFrame with bounding box coordinates
            dataset: Full image dataset
            
        Returns:
            DataFrame with added crop column
        """
        df = bounds_df.copy()
        df['crop'] = df[['y0', 'y1', 'x0', 'x1']].apply(
            lambda x: self.produce_crop(dataset, *x), axis=1
        )
        return df
