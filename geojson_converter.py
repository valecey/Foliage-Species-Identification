"""
GeoJSON conversion utilities for species detection pipeline.
Converts Label Studio annotations to GeoJSON format with proper geographic coordinates.
"""
import json
import rasterio
from rasterio.warp import transform_bounds
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path


class GeoJSONConverter:
    """Converts Label Studio annotations to GeoJSON format."""
    
    @staticmethod
    def extract_geographic_bounds(tif_path: str) -> Tuple[float, float, float, float]:
        """
        Extract geographic bounds from a GeoTIFF file.
        
        Args:
            tif_path: Path to the GeoTIFF file
            
        Returns:
            Tuple of (lon_min, lon_max, lat_min, lat_max) in WGS84 coordinates
        """
        with rasterio.open(tif_path) as src:
            # Get the CRS information
            crs = src.crs
            epsg_code = crs.to_epsg()
            
            # Get the bounds in the original coordinate system
            bounds = src.bounds
            x_min = bounds.left
            x_max = bounds.right
            y_min = bounds.bottom
            y_max = bounds.top
            
            # Check if it's already WGS84 (EPSG:4326)
            if epsg_code == 4326:
                lon_min, lat_min, lon_max, lat_max = x_min, y_min, x_max, y_max
            else:
                # Convert to WGS84
                wgs84_bounds = transform_bounds(crs, "EPSG:4326", x_min, y_min, x_max, y_max)
                lon_min, lat_min, lon_max, lat_max = wgs84_bounds
            
            return lon_min, lon_max, lat_min, lat_max
    
    @staticmethod
    def normalize_coordinates_to_geojson(points: List[List[float]], 
                                     lon_min: float, lon_max: float, 
                                     lat_min: float, lat_max: float) -> List[List[float]]:
        """
        Convert normalized coordinates (0-100) to geographical coordinates.
        
        Args:
            points: List of [x, y] coordinates in normalized format
            lon_min: Minimum longitude boundary
            lon_max: Maximum longitude boundary
            lat_min: Minimum latitude boundary
            lat_max: Maximum latitude boundary
            
        Returns:
            List of [longitude, latitude] coordinates for GeoJSON
        """
        geojson_points = []
        lon_range = lon_max - lon_min
        lat_range = lat_max - lat_min
        
        for point in points:
            # Convert from normalized (0-100) to geo coordinates
            x, y = point
            lon = lon_min + (x / 100.0) * lon_range
            # Invert Y axis as GeoJSON uses different orientation
            lat = lat_max - (y / 100.0) * lat_range
            
            geojson_points.append([lon, lat])
        
        return geojson_points
    
    @staticmethod
    def normalize_rectangle_to_geojson_polygon(rect: Dict[str, float],
                                     lon_min: float, lon_max: float, 
                                     lat_min: float, lat_max: float) -> List[List[float]]:
        """
        Convert normalized rectangle coordinates (0-100) to geographical polygon coordinates.
        
        Args:
            rect: Dictionary with x, y, width, height values in normalized format
            lon_min: Minimum longitude boundary
            lon_max: Maximum longitude boundary
            lat_min: Minimum latitude boundary
            lat_max: Maximum latitude boundary
            
        Returns:
            List of [longitude, latitude] coordinates for GeoJSON polygon
        """
        # Extract rectangle coordinates
        x = rect.get('x', 0)
        y = rect.get('y', 0)
        width = rect.get('width', 0)
        height = rect.get('height', 0)
        
        # Create the four corners of the rectangle (in normalized coordinates)
        points = [
            [x, y],                  # top-left
            [x + width, y],          # top-right
            [x + width, y + height], # bottom-right
            [x, y + height],         # bottom-left
            [x, y]                   # close the polygon
        ]
        
        # Convert to GeoJSON coordinates
        geojson_points = []
        lon_range = lon_max - lon_min
        lat_range = lat_max - lat_min
        
        for point in points:
            x_norm, y_norm = point
            lon = lon_min + (x_norm / 100.0) * lon_range
            # Invert Y axis as GeoJSON uses different orientation
            lat = lat_max - (y_norm / 100.0) * lat_range
            
            geojson_points.append([lon, lat])
        
        return geojson_points
    
    @staticmethod
    def label_studio_to_geojson(label_studio_data: List[Dict], 
                                target_id: Optional[int] = None,
                                lon_bounds: Tuple[float, float] = (103.825, 103.826),
                                lat_bounds: Tuple[float, float] = (1.372, 1.373)) -> Dict:
        """
        Convert Label Studio annotations (polygon or rectangle) to GeoJSON format.
        
        Args:
            label_studio_data: Data from Label Studio export
            target_id: Optional ID to filter for a specific annotation
            lon_bounds: Tuple of (min_longitude, max_longitude) for conversion
            lat_bounds: Tuple of (min_latitude, max_latitude) for conversion
            
        Returns:
            GeoJSON FeatureCollection
        """
        lon_min, lon_max = lon_bounds
        lat_min, lat_max = lat_bounds
        
        features = []
        
        for item in label_studio_data:
            if not item.get('annotations'):
                continue

            # Check if this item has the specific ID we're looking for
            item_id = item.get('id')
            if target_id is not None and item_id != target_id:
                continue
                
            # Get file name to use as identifier
            file_name = item.get('file_upload', '').split('/')[-1]
            
            for annotation in item['annotations']:
                for result in annotation.get('result', []):
                    result_type = result.get('type')
                    
                    # Handle polygon annotations
                    if result_type == 'polygonlabels':
                        points = result['value'].get('points', [])
                        label = result['value'].get('polygonlabels', ['Unknown'])[0]
                        
                        # Convert to GeoJSON coordinates
                        geojson_points = GeoJSONConverter.normalize_coordinates_to_geojson(
                            points, lon_min, lon_max, lat_min, lat_max
                        )
                        
                        # Close the polygon if not already closed
                        if geojson_points[0] != geojson_points[-1]:
                            geojson_points.append(geojson_points[0])
                    
                    # Handle rectangle annotations
                    elif result_type == 'rectanglelabels':
                        rect_value = result['value']
                        label = rect_value.get('rectanglelabels', ['Unknown'])[0]
                        vote_count = rect_value.get('vote_count', 0)
                        
                        # Convert rectangle to polygon points in GeoJSON format
                        geojson_points = GeoJSONConverter.normalize_rectangle_to_geojson_polygon(
                            rect_value, lon_min, lon_max, lat_min, lat_max
                        )
                    
                    else:
                        # Skip unsupported annotation types
                        continue
                    
                    # Create GeoJSON feature
                    properties = {
                        "label": label,
                        "source_file": file_name,
                        "annotation_id": annotation['id'],
                        "result_id": result['id']
                    }
                    
                    # Add vote count for rectangle labels if available
                    if result_type == 'rectanglelabels' and 'vote_count' in result['value']:
                        properties["vote_count"] = result['value']['vote_count']
                    
                    feature = {
                        "type": "Feature",
                        "properties": properties,
                        "geometry": {
                            "type": "MultiPolygon",
                            "coordinates": [[geojson_points]]
                        }
                    }
                    
                    features.append(feature)
        
        # Create GeoJSON FeatureCollection
        geojson = {
            "type": "FeatureCollection",
            "name": "label_studio_annotations",
            "crs": {
                "type": "name", 
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
            },
            "features": features
        }
        
        return geojson
    
    @staticmethod
    def convert_and_save_geojson(label_studio_data: List[Dict],
                                 output_path: str,
                                 tif_path: Optional[str] = None,
                                 target_id: Optional[int] = None,
                                 lon_bounds: Optional[Tuple[float, float]] = None,
                                 lat_bounds: Optional[Tuple[float, float]] = None) -> Dict:
        """
        Convert Label Studio data to GeoJSON and save to file.
        
        Args:
            label_studio_data: Label Studio annotation data
            output_path: Output GeoJSON file path
            tif_path: Optional path to TIF file for automatic bounds extraction
            target_id: Optional ID to filter for a specific annotation
            lon_bounds: Manual longitude bounds (overrides tif_path)
            lat_bounds: Manual latitude bounds (overrides tif_path)
            
        Returns:
            GeoJSON FeatureCollection
        """
        # Extract bounds from TIF file if provided and manual bounds not specified
        if tif_path and Path(tif_path).exists() and not (lon_bounds and lat_bounds):
            lon_min, lon_max, lat_min, lat_max = GeoJSONConverter.extract_geographic_bounds(tif_path)
            lon_bounds = (lon_min, lon_max)
            lat_bounds = (lat_min, lat_max)
            print(f"Extracted bounds from {tif_path}:")
            print(f"  Longitude: {lon_min:.6f} to {lon_max:.6f}")
            print(f"  Latitude: {lat_min:.6f} to {lat_max:.6f}")
        
        # Use default bounds if none provided
        if not lon_bounds or not lat_bounds:
            lon_bounds = (103.825, 103.826)
            lat_bounds = (1.372, 1.373)
            print("Using default geographic bounds")
        
        # Convert to GeoJSON
        geojson = GeoJSONConverter.label_studio_to_geojson(
            label_studio_data,
            target_id=target_id,
            lon_bounds=lon_bounds,
            lat_bounds=lat_bounds
        )
        
        # Save to file
        with open(output_path, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print(f"Converted {len(geojson['features'])} features to GeoJSON")
        print(f"Saved to: {output_path}")
        
        return geojson
