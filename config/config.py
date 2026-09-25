"""
Configuration module for species detection pipeline.
Provides backward compatibility while using new schema-based system.
"""
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

from .schema import SpeciesDetectionConfig
from .loader import load_config


class Config:
    """
    Backward-compatible configuration class.
    
    This class maintains the same interface as the original Config class
    while using the new schema-based configuration system internally.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None, **kwargs):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file with overrides.
            **kwargs: Additional configuration overrides.
        """
        # Load configuration using the new system
        self._config = load_config(config_path, **kwargs)
    
    # Delegate all attribute access to the underlying configuration
    def __getattr__(self, name):
        """Delegate attribute access to the underlying configuration."""
        return getattr(self._config, name.upper(), getattr(self._config, name))
    
    @property
    def _config_instance(self) -> SpeciesDetectionConfig:
        """Access to the underlying configuration instance."""
        return self._config
    
    @classmethod
    def get_alpha_values(cls, c_values: List[float]) -> List[float]:
        """Convert C values to alpha values for SGD."""
        return [1/i for i in c_values]
    
    @classmethod
    def get_class_weights(cls, w_c_values: List[float]) -> List[Dict[int, float]]:
        """Generate class weight configurations."""
        return [{0: i, 1: 1-i} for i in w_c_values]
    
    def get_title_dict(self) -> Dict[str, str]:
        """Get capture type to title mapping."""
        return self._config.get_title_dict()