"""
Configuration loader that merges defaults with YAML overrides.
"""
import yaml
from pathlib import Path
from typing import Optional, Union, Dict, Any
from .schema import SpeciesDetectionConfig


class ConfigLoader:
    """Loads and merges configuration from YAML files with schema validation."""
    
    def __init__(self, default_config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration loader.
        
        Args:
            default_config_path: Path to default configuration file.
                               If None, uses the built-in default.
        """
        if default_config_path is None:
            self.default_config_path = Path(__file__).parent / "default_config.yaml"
        else:
            self.default_config_path = Path(default_config_path)
    
    def load_config(
        self, 
        config_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> SpeciesDetectionConfig:
        """
        Load configuration from YAML file and merge with defaults.
        
        Args:
            config_path: Path to configuration file with overrides.
                        If None, uses only defaults.
            **kwargs: Additional configuration overrides as keyword arguments.
            
        Returns:
            Validated SpeciesDetectionConfig instance.
            
        Raises:
            FileNotFoundError: If default config file doesn't exist.
            ValidationError: If configuration doesn't match schema.
        """
        # Load default configuration
        if not self.default_config_path.exists():
            raise FileNotFoundError(
                f"Default configuration file not found: {self.default_config_path}"
            )
        
        with open(self.default_config_path, 'r') as f:
            default_config = yaml.safe_load(f)
        
        # Load override configuration if provided
        override_config = {}
        if config_path is not None:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            with open(config_path, 'r') as f:
                override_config = yaml.safe_load(f)
        
        # Merge configurations
        merged_config = self._merge_configs(default_config, override_config)
        
        # Apply kwargs overrides
        if kwargs:
            merged_config = self._apply_kwargs_overrides(merged_config, kwargs)
        
        # Validate and return configuration
        return SpeciesDetectionConfig(**merged_config)
    
    def _merge_configs(self, default: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge override configuration with default configuration.
        
        Args:
            default: Default configuration dictionary.
            override: Override configuration dictionary.
            
        Returns:
            Merged configuration dictionary.
        """
        merged = default.copy()
        
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _apply_kwargs_overrides(self, config: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply keyword argument overrides to configuration.
        
        Supports dot notation for nested keys (e.g., 'dataset.name').
        
        Args:
            config: Configuration dictionary.
            kwargs: Keyword argument overrides.
            
        Returns:
            Updated configuration dictionary.
        """
        result = config.copy()
        
        for key, value in kwargs.items():
            if '.' in key:
                # Handle nested keys with dot notation
                keys = key.split('.')
                current = result
                
                # Navigate to the parent of the target key
                for k in keys[:-1]:
                    if k not in current:
                        current[k] = {}
                    current = current[k]
                
                # Set the final value
                current[keys[-1]] = value
            else:
                # Handle top-level keys
                result[key] = value
        
        return result
    
    def save_config(
        self, 
        config: SpeciesDetectionConfig, 
        output_path: Union[str, Path]
    ) -> None:
        """
        Save configuration to YAML file.
        
        Args:
            config: Configuration instance to save.
            output_path: Path where to save the configuration.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict and save
        config_dict = config.dict()
        
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)


# Global configuration loader instance
_config_loader = ConfigLoader()


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> SpeciesDetectionConfig:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to configuration file with overrides.
        **kwargs: Additional configuration overrides.
        
    Returns:
        Validated SpeciesDetectionConfig instance.
    """
    return _config_loader.load_config(config_path, **kwargs)


def save_config(
    config: SpeciesDetectionConfig, 
    output_path: Union[str, Path]
) -> None:
    """
    Convenience function to save configuration.
    
    Args:
        config: Configuration instance to save.
        output_path: Path where to save the configuration.
    """
    _config_loader.save_config(config, output_path)
