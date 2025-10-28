"""Utilities for model size presets and benchmarking configuration.

This module provides dataclasses to hold transformer model size parameters
and benchmarking configuration, along with convenience functions to manage them.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional
import torch


@dataclass(frozen=True)
class ModelSizeConfig:
    """Configuration for a model size preset.

    This is an immutable configuration class - all attributes are frozen
    after creation to prevent accidental modifications.

    Attributes:
        size: human-readable size name (e.g. 'small', '2.7B')
        d_model: model hidden dimension
        d_ff: feed-forward hidden dimension
        num_layers: number of transformer layers
        num_heads: number of attention heads
    """

    size: str
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


_PRESETS: Dict[str, ModelSizeConfig] = {
    "small": ModelSizeConfig("small", d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelSizeConfig("medium", d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelSizeConfig("large", d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelSizeConfig("xl", d_model=1600, d_ff=6400, num_layers=48, num_heads=25),
    "2.7b": ModelSizeConfig("2.7B", d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
}


_ALIASES: Dict[str, str] = {
    "xlarge": "xl",
    "2.7": "2.7b",
    "2.7b": "2.7b",
    "2_7b": "2.7b",
    "2_7": "2.7b",
}


def _normalize_key(key: str) -> str:
    if not key:
        return ""
    k = key.strip().lower().replace(" ", "").replace("-", "").replace("gb", "b")
    # common aliases
    if k in _PRESETS:
        return k
    if k in _ALIASES:
        return _ALIASES[k]
    # handle variants like '2.7b', '2.7', '2_7b'
    k2 = k.replace("_", "").replace(".", "")
    if k2.startswith("27"):
        return "2.7b"
    return k


def get_model_config(size: str) -> ModelSizeConfig:
    """Return a ModelSizeConfig for the given size name.

    Accepts case-insensitive names and common aliases. Known names:
    'small', 'medium', 'large', 'xl' (or 'xlarge'), and '2.7B' (also '2.7', '2_7b').

    Raises:
        ValueError: if the provided size name is unknown.
    """
    key = _normalize_key(size)
    if key in _PRESETS:
        return _PRESETS[key]
    raise ValueError(f"Unknown model size: {size!r}. Known sizes: {', '.join(sorted(_PRESETS.keys()))}")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for benchmarking runs.
    
    This is an immutable configuration class - all attributes are frozen
    after creation to prevent accidental modifications. To create a new
    configuration with different values, create a new instance:
    
    Example:
        cfg_dict = vars(old_config)
        cfg_dict['model_size'] = 'large'
        new_config = BenchmarkConfig(**cfg_dict)
    """
    # Model configuration
    model_size: str = ""  # If provided, overrides d_model/num_layers/etc
    vocab_size: int = 10_000
    context_length: int = 256
    d_model: int = 768
    num_layers: int = 12
    num_heads: int = 12
    d_ff: int = 3072
    rope_theta: float = 10000.0
    
    # Data generation
    batch_size: int = 4
    
    # Runtime configuration
    device: str = "mps"
    dtype: str = "float32"
    warm_up_steps: int = 5
    n_steps: int = 10

    def __post_init__(self):
        """Apply model size preset if specified."""
        if self.model_size:
            cfg = get_model_config(self.model_size)
            # Use object.__setattr__ to set frozen attributes during initialization
            object.__setattr__(self, 'd_model', cfg.d_model)
            object.__setattr__(self, 'num_layers', cfg.num_layers)
            object.__setattr__(self, 'num_heads', cfg.num_heads)
            object.__setattr__(self, 'd_ff', cfg.d_ff)

    @classmethod
    def from_args(cls, args) -> "BenchmarkConfig":
        """Create config from argparse namespace."""
        return cls(**{k: v for k, v in vars(args).items() 
                     if k in BenchmarkConfig.__dataclass_fields__})

    def to_device(self) -> torch.device:
        """Get torch device instance."""
        if self.device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("MPS (Apple metal chip) backend not available.")
        return torch.device(self.device)


__all__ = ["ModelSizeConfig", "get_model_config", "BenchmarkConfig"]
