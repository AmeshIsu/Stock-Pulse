"""Configuration loader module for Stock Pulse."""
from pathlib import Path
import yaml
import json
from typing import Any, Dict

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"

def load_config(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Load and return configuration from config.yaml."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def load_thresholds(thresholds_path: str | Path | None = None) -> Dict[str, Any]:
    """Load and return thresholds definitions from thresholds.json."""
    if thresholds_path is None:
        cfg = load_config()
        base_dir = Path(__file__).resolve().parents[2]
        thresholds_path = base_dir / cfg["paths"]["thresholds_file"]
    
    path = Path(thresholds_path)
    if not path.exists():
        raise FileNotFoundError(f"Thresholds file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        thresholds = json.load(f)
    return thresholds
