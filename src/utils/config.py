"""
RetailPulse Configuration Loader
Safely loads and validates configuration parameters from config.yaml.
"""

import os
from typing import Any, Dict
import yaml

_CONFIG_CACHE: Dict[str, Any] = {}

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Loads configuration settings from a YAML file.
    
    Args:
        config_path: Path to the configuration YAML file.
        
    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE and os.path.exists(config_path):
        return _CONFIG_CACHE

    if not os.path.exists(config_path):
        # Fallback to parent dir search
        alt_path = os.path.join(os.path.dirname(__file__), "..", "..", config_path)
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    _CONFIG_CACHE = config
    return config

def get_path(key_path: str, default: str = "") -> str:
    """
    Retrieves a nested path from configuration.
    Example: get_path('paths.raw_data_dir')
    """
    cfg = load_config()
    keys = key_path.split(".")
    val = cfg
    for k in keys:
        if isinstance(val, dict) and k in val:
            val = val[k]
        else:
            return default
    return str(val)
