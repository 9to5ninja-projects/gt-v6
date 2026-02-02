"""
GroundThink v7 - GDN + SWA Hybrid Architecture

Clean break from v6:
- Chunk-recurrent Delta Rule (numerically stable)
- Modular package structure (no more 2000-line notebooks)
- Proper gradient checkpointing

Usage:
    from groundthink_v7 import HybridConfig, TransparentHybrid
    from groundthink_v7 import train_curriculum, proper_niah_test
    
    cfg = HybridConfig(d_model=256, layer_pattern="GS")
    model = TransparentHybrid(cfg).cuda()
    
    # Train
    data_loader = load_wikitext()
    train_curriculum(model, data_loader, steps=1000)
    
    # Evaluate
    proper_niah_test(model)
"""

__version__ = "7.0.0"

from .config import HybridConfig
from .model import (
    GatedDeltaNetLayer,
    SlidingWindowAttention, 
    TransparentHybrid,
    RMSNorm,
    SwiGLUFFN,
)
from .analysis import (
    proper_niah_test,
    test_niah_by_distance,
    run_full_diagnostic,
    validate_delta_rule,
    train_curriculum,
    analyze_gradients,
    TextDataset,
    load_wikitext,
)
from .core import chunk_delta_rule, CHUNK_SIZE

__all__ = [
    # Config
    'HybridConfig',
    
    # Model
    'GatedDeltaNetLayer',
    'SlidingWindowAttention',
    'TransparentHybrid',
    'RMSNorm',
    'SwiGLUFFN',
    
    # Core ops
    'chunk_delta_rule',
    'CHUNK_SIZE',
    
    # Analysis
    'proper_niah_test',
    'test_niah_by_distance', 
    'run_full_diagnostic',
    'validate_delta_rule',
    'train_curriculum',
    'analyze_gradients',
    'TextDataset',
    'load_wikitext',
]
