# -*- coding: utf-8 -*-
"""GroundThink V6 - Hybrid GatedDeltaNet + SWAttention

Clean architecture using FLA library components directly.
Lessons from V5: identity protection, rare token monitoring, gradient health.
New: configurable layer patterns, proper masking, gradient checkpointing.

Run on Colab with A100/L4 GPU.

DEPENDENCIES (run these first):
    pip install triton
    pip install flash-linear-attention
    pip install datasets transformers
    pip install vllm
"""

# ============================================================================
# CELL 0: ENVIRONMENT SETUP
# ============================================================================

# Uncomment for Colab:
# from google.colab import drive
# drive.mount('/content/drive')
# !pip install -q triton
# !pip install -q flash-linear-attention
# !pip install -q datasets transformers
# !pip install -q vllm

import sys
print(f"Python: {sys.version}")

# ============================================================================
# CELL 1: CONFIGURATION (SINGLE SOURCE OF TRUTH)
# ============================================================================
from dataclasses import dataclass, field
from typing import List, Optional
import torch

@dataclass
class ModelConfig:
    """All model hyperparameters in one place."""
    # Architecture
    vocab_size: int = 50257          # GPT-2 tokenizer
    d_model: int = 512
    n_layers: int = 12
    n_heads: int = 8                  # For SWA layers
    head_dim: int = 64                # d_model // n_heads
    
    # Hybrid pattern: SWA every N layers (e.g., 4 = 3:1 Mamba:SWA ratio)
    attn_interval: int = 4            # SWA at layers 3, 7, 11 for 12-layer model
    
    # Sliding window attention
    window_size: int = 2048           # Increased from V5's 512
    
    # GatedDeltaNet (Mamba-2 equivalent) settings
    expand_k: float = 1.0
    expand_v: float = 2.0
    
    # Training
    max_seq_len: int = 2048
    use_gradient_checkpointing: bool = True  # Checkpoint SWA layers
    
    # Weight tying
    tie_weights: bool = True
    
    def __post_init__(self):
        assert self.d_model % self.n_heads == 0
        self.head_dim = self.d_model // self.n_heads
        
    def get_swa_layer_indices(self) -> List[int]:
        """Returns which layer indices should have SWA."""
        # Pattern: SWA at (interval-1), (2*interval-1), etc.
        # For interval=4, n_layers=12: layers 3, 7, 11
        return [i for i in range(self.n_layers) 
                if i % self.attn_interval == (self.attn_interval - 1)]


@dataclass 
class TrainConfig:
    """Training hyperparameters."""
    # Data
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    dataset_subset: str = "sample-10BT"
    target_tokens: int = 20_000_000
    
    # Batch
    batch_size: int = 2
    seq_len: int = 512
    accum_steps: int = 2
    
    # Optimization
    steps: int = 10000
    warmup_ratio: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    betas: tuple = (0.9, 0.95)
    
    # Regularization (identity protection for rare tokens)
    dt_reg_lambda: float = 0.0       # Set >0 to enable (experimental with FLA)
    dt_reg_threshold: int = 30000
    
    # Monitoring
    log_interval: int = 50
    grad_log_interval: int = 500
    niah_checkpoints: List[int] = field(default_factory=lambda: [
        500, 1000, 2000, 3000, 5000, 7500, 10000
    ])
    
    @property
    def warmup_steps(self) -> int:
        return int(self.steps * self.warmup_ratio)
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.accum_steps


# Create default configs (modify these for experiments)
MODEL_CFG = ModelConfig()
TRAIN_CFG = TrainConfig()

print("=" * 60)
print("GROUNDTHINK V6 CONFIGURATION")
print("=" * 60)
print(f"\n[MODEL]")
print(f"  d_model:        {MODEL_CFG.d_model}")
print(f"  n_layers:       {MODEL_CFG.n_layers}")
print(f"  n_heads:        {MODEL_CFG.n_heads} (head_dim={MODEL_CFG.head_dim})")
print(f"  attn_interval:  {MODEL_CFG.attn_interval} (SWA at layers {MODEL_CFG.get_swa_layer_indices()})")
print(f"  window_size:    {MODEL_CFG.window_size}")
print(f"  weight_tying:   {MODEL_CFG.tie_weights}")
print(f"  grad_ckpt:      {MODEL_CFG.use_gradient_checkpointing}")
print(f"\n[TRAINING]")
print(f"  batch_size:     {TRAIN_CFG.batch_size} x {TRAIN_CFG.accum_steps} = {TRAIN_CFG.effective_batch_size}")
print(f"  seq_len:        {TRAIN_CFG.seq_len}")
print(f"  steps:          {TRAIN_CFG.steps}")
print(f"  lr:             {TRAIN_CFG.lr}")
print(f"  warmup:         {TRAIN_CFG.warmup_steps} steps")

# ============================================================================
# CELL 2: CORE IMPORTS & DEVICE
# ============================================================================
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoTokenizer

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f"\n✓ GPU: {props.name} ({props.total_memory/1e9:.1f}GB)")
    # Enable TF32 for faster training on Ampere+
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# FLA imports
try:
    from fla.layers import GatedDeltaNet
    print("✓ GatedDeltaNet imported")
except ImportError as e:
    print(f"✗ GatedDeltaNet import failed: {e}")
    raise

try:
    # Try primary import path
    from fla.layers import SlidingWindowAttention as FLA_SWA
    print("✓ SlidingWindowAttention imported")
except ImportError:
    try:
        # Alternate import path
        from fla.modules import SlidingWindowAttention as FLA_SWA
        print("✓ SlidingWindowAttention imported (from fla.modules)")
    except ImportError as e:
        print(f"✗ SlidingWindowAttention import failed: {e}")
        print("  Checking available FLA modules...")
        import fla
        print(f"  fla.layers contains: {dir(fla.layers)}")
        raise

FLA_AVAILABLE = True
print("✓ FLA library loaded")

# ============================================================================
# CELL 3: MODEL COMPONENTS
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class SwiGLUFFN(nn.Module):
    """SwiGLU Feed-Forward Network."""
    def __init__(self, d_model: int, expansion: float = 8/3):
        super().__init__()
        hidden = int(d_model * expansion)
        hidden = ((hidden + 63) // 64) * 64  # Align to 64 for efficiency
        
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)
        self.norm = RMSNorm(d_model)
        
        # Initialize
        nn.init.normal_(self.w1.weight, std=0.02)
        nn.init.normal_(self.w3.weight, std=0.02)
        nn.init.normal_(self.w2.weight, std=0.02 / math.sqrt(2))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        return residual + self.w2(F.silu(self.w1(x)) * self.w3(x))


# ============================================================================
# CELL 4: HYBRID MODEL
# ============================================================================

class HybridBlock(nn.Module):
    """
    A single layer that can be either:
    - GatedDeltaNet (Mamba-2 equivalent) for efficient long-range modeling
    - SlidingWindowAttention for precise local/retrieval tasks
    
    Both wrapped with pre-norm and residual.
    Supports use_cache for inference (vLLM IsHybrid protocol).
    """
    def __init__(
        self,
        d_model: int,
        is_attention: bool,
        n_heads: int = 8,
        window_size: int = 2048,
        expand_k: float = 1.0,
        expand_v: float = 2.0,
        layer_idx: int = 0,
    ):
        super().__init__()
        self.is_attention = is_attention
        self.layer_idx = layer_idx
        self.norm = RMSNorm(d_model)
        
        if is_attention:
            if FLA_AVAILABLE:
                self.layer = FLA_SWA(
                    hidden_size=d_model,
                    num_heads=n_heads,
                    window_size=window_size,
                    layer_idx=layer_idx,
                )
            else:
                raise NotImplementedError("FLA required for SWA")
        else:
            if FLA_AVAILABLE:
                self.layer = GatedDeltaNet(
                    hidden_size=d_model,
                    expand_k=expand_k,
                    expand_v=expand_v,
                    layer_idx=layer_idx,
                )
            else:
                raise NotImplementedError("FLA required for GatedDeltaNet")
    
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None,
        past_state: Optional[torch.Tensor] = None,
        use_cache: bool = False,
    ) -> tuple:
        """
        Args:
            x: [B, T, D] input
            attention_mask: optional mask for SWA
            past_state: recurrent state (GDN) or KV cache (SWA)
            use_cache: whether to return updated state
        
        Returns:
            output: [B, T, D]
            new_state: updated state if use_cache, else None
        """
        residual = x
        x = self.norm(x)
        
        new_state = None
        
        if use_cache:
            # FLA layers support state passing
            if self.is_attention:
                x, new_state = self.layer(x, attention_mask=attention_mask, 
                                          past_key_values=past_state, use_cache=True)
            else:
                x, new_state = self.layer(x, past_state=past_state, use_cache=True)
        else:
            # Training mode - no state tracking
            if self.is_attention and attention_mask is not None:
                x = self.layer(x, attention_mask=attention_mask)
            else:
                x = self.layer(x)
        
        return residual + x, new_state


class GroundThinkLM(nn.Module):
    """
    Hybrid Language Model: GatedDeltaNet + SlidingWindowAttention
    
    Architecture:
    - Embedding with optional weight tying
    - N layers of HybridBlock (mix of GDN and SWA based on attn_interval)
    - FFN after each block
    - Final RMSNorm + LM head
    
    Designed for vLLM IsHybrid protocol compatibility.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        
        # Embedding
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embed.weight, std=0.02)
        
        # Determine which layers get attention
        swa_indices = set(cfg.get_swa_layer_indices())
        
        # Build layers
        self.blocks = nn.ModuleList()
        self.ffns = nn.ModuleList()
        
        for i in range(cfg.n_layers):
            is_attn = i in swa_indices
            self.blocks.append(HybridBlock(
                d_model=cfg.d_model,
                is_attention=is_attn,
                n_heads=cfg.n_heads,
                window_size=cfg.window_size,
                expand_k=cfg.expand_k,
                expand_v=cfg.expand_v,
                layer_idx=i,
            ))
            self.ffns.append(SwiGLUFFN(cfg.d_model))
        
        # Output
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        
        # Weight tying
        if cfg.tie_weights:
            self.lm_head.weight = self.embed.weight
        else:
            nn.init.normal_(self.lm_head.weight, std=0.02)
        
        # Track which layers are attention for checkpointing and state management
        self._swa_indices = swa_indices
        self._gdn_indices = set(range(cfg.n_layers)) - swa_indices
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_states: Optional[List[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple:
        """
        Args:
            input_ids: [B, T] token indices
            targets: [B, T] target token indices (optional, for loss)
            attention_mask: [B, T] or [B, 1, T, T] (optional)
            past_states: list of per-layer states for inference
            use_cache: whether to return updated states
        
        Returns:
            logits: [B, T, V]
            loss: scalar if targets provided, else None
            new_states: list of states if use_cache, else None
        """
        x = self.embed(input_ids)
        
        new_states = [] if use_cache else None
        
        for i, (block, ffn) in enumerate(zip(self.blocks, self.ffns)):
            past_state = past_states[i] if past_states is not None else None
            
            # Gradient checkpointing for SWA layers (memory hungry) - training only
            if self.cfg.use_gradient_checkpointing and self.training and i in self._swa_indices:
                x = checkpoint(
                    self._forward_block_train, 
                    block, ffn, x, attention_mask,
                    use_reentrant=False
                )
            else:
                x, state = block(x, attention_mask, past_state, use_cache)
                x = ffn(x)
                if use_cache:
                    new_states.append(state)
        
        x = self.norm_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), 
                targets.view(-1),
                ignore_index=-100
            )
        
        return logits, loss, new_states
    
    @staticmethod
    def _forward_block_train(block, ffn, x, attention_mask):
        """Helper for gradient checkpointing (training only, no cache)."""
        x, _ = block(x, attention_mask, None, False)
        x = ffn(x)
        return x
    
    # =========================================================================
    # vLLM IsHybrid Protocol Methods
    # =========================================================================
    
    @classmethod
    def get_state_dtype_from_config(cls, config: ModelConfig) -> dict:
        """Return dtype for each state type (vLLM requirement)."""
        return {
            'gdn': torch.bfloat16,  # GatedDeltaNet recurrent state
            'swa': torch.bfloat16,  # SWA KV cache
        }
    
    @classmethod
    def get_state_shape_from_config(cls, config: ModelConfig, batch_size: int = 1) -> dict:
        """
        Return state shapes for vLLM pre-allocation (vLLM requirement).
        
        GatedDeltaNet state: depends on expand_k, expand_v, hidden_size
        SWA KV cache: [batch, n_heads, window_size, head_dim] for K and V
        """
        head_dim = config.d_model // config.n_heads
        
        # GDN state shape (from FLA's GatedDeltaNet)
        # This is approximate - actual shape depends on FLA internals
        gdn_state_dim = int(config.d_model * config.expand_v)
        
        return {
            'gdn': (batch_size, gdn_state_dim),
            'swa_k': (batch_size, config.n_heads, config.window_size, head_dim),
            'swa_v': (batch_size, config.n_heads, config.window_size, head_dim),
        }
    
    def get_layer_types(self) -> List[str]:
        """Return layer type for each layer (for vLLM KVCacheGroups)."""
        return ['swa' if i in self._swa_indices else 'gdn' 
                for i in range(self.cfg.n_layers)]
    
    def count_parameters(self) -> dict:
        """Count parameters by component."""
        counts = {
            'embed': sum(p.numel() for p in self.embed.parameters()),
            'gdn_layers': 0,
            'swa_layers': 0,
            'ffn_layers': 0,
            'lm_head': 0 if self.cfg.tie_weights else sum(p.numel() for p in self.lm_head.parameters()),
        }
        
        for i, (block, ffn) in enumerate(zip(self.blocks, self.ffns)):
            block_params = sum(p.numel() for p in block.parameters())
            ffn_params = sum(p.numel() for p in ffn.parameters())
            
            if i in self._swa_indices:
                counts['swa_layers'] += block_params
            else:
                counts['gdn_layers'] += block_params
            counts['ffn_layers'] += ffn_params
        
        counts['total'] = sum(counts.values())
        return counts


# ============================================================================
# CELL 5: MONITORING & DIAGNOSTICS (FROM V5)
# ============================================================================

def log_gradient_norms(model: nn.Module) -> dict:
    """Log gradient norms per component."""
    norms = {}
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            norms[name] = param.grad.norm().item()
    
    return norms


def print_gradient_summary(model: nn.Module):
    """Print condensed gradient summary by layer type."""
    norms = log_gradient_norms(model)
    
    # Aggregate by component type
    aggregated = {
        'embed': [],
        'gdn': [],
        'swa': [],
        'ffn': [],
        'norm': [],
        'lm_head': []
    }
    
    swa_indices = model._swa_indices if hasattr(model, '_swa_indices') else set()
    
    for name, norm in norms.items():
        if 'embed' in name:
            aggregated['embed'].append(norm)
        elif 'lm_head' in name:
            aggregated['lm_head'].append(norm)
        elif 'norm' in name:
            aggregated['norm'].append(norm)
        elif 'ffn' in name:
            aggregated['ffn'].append(norm)
        elif 'blocks' in name:
            # Extract layer index
            parts = name.split('.')
            if len(parts) >= 2:
                try:
                    layer_idx = int(parts[1])
                    if layer_idx in swa_indices:
                        aggregated['swa'].append(norm)
                    else:
                        aggregated['gdn'].append(norm)
                except ValueError:
                    pass
    
    print("\n📊 Gradient Norms:")
    print("-" * 50)
    for component, vals in aggregated.items():
        if vals:
            mean = np.mean(vals)
            max_v = np.max(vals)
            bar = "█" * int(min(max_v * 5, 20))
            flag = " ⚠️" if max_v > 5.0 else ""
            print(f"  {component:<10} mean={mean:6.3f} max={max_v:6.2f} {bar}{flag}")


def needle_test(
    model: nn.Module, 
    tokenizer,
    seq_len: int = 512, 
    n_trials: int = 50,
    needle_token: int = 50250,
    device: str = "cuda"
) -> dict:
    """
    Needle-in-a-Haystack test for retrieval capability.
    
    Places a rare token at a random position and checks if the model
    can "remember" it at the final position (predicts it with higher prob
    than random chance).
    """
    model.eval()
    random_chance = 1.0 / tokenizer.vocab_size
    
    probs = []
    with torch.no_grad():
        for _ in range(n_trials):
            # Random haystack (common tokens)
            tokens = torch.randint(1000, 10000, (1, seq_len), device=device)
            
            # Insert needle at random position (avoid edges)
            pos = torch.randint(64, seq_len - 64, (1,)).item()
            tokens[0, pos] = needle_token
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, _, _ = model(tokens)  # 3-tuple return
            
            # Check probability of needle at final position
            p = F.softmax(logits[0, -1].float(), dim=-1)[needle_token].item()
            probs.append(p)
    
    return {
        'mean': np.mean(probs),
        'std': np.std(probs),
        'random_chance': random_chance,
        'ratio': np.mean(probs) / random_chance,
    }


def probe_layer_representations(
    model: nn.Module,
    tokenizer,
    needle_id: int = 50250,
    seq_len: int = 512,
    needle_pos: int = 256,
    device: str = "cuda"
):
    """
    Track how the needle's representation evolves through layers.
    Uses cosine similarity to original embedding.
    """
    model.eval()
    
    # Create test sequence
    tokens = torch.randint(1000, 10000, (1, seq_len), device=device)
    tokens[0, needle_pos] = needle_id
    
    with torch.no_grad():
        x = model.embed(tokens)
        needle_embed = model.embed.weight[needle_id].float()
        
        print(f"\n📍 Layer-wise needle (token {needle_id}) representation:")
        print("-" * 50)
        
        swa_indices = model._swa_indices if hasattr(model, '_swa_indices') else set()
        
        for i, (block, ffn) in enumerate(zip(model.blocks, model.ffns)):
            x, _ = block(x, attention_mask=None, past_state=None, use_cache=False)
            x = ffn(x)
            
            needle_hidden = x[0, needle_pos].float()
            sim = F.cosine_similarity(needle_hidden, needle_embed, dim=0).item()
            
            layer_type = "SWA" if i in swa_indices else "GDN"
            bar = "█" * int(max(0, (sim + 1) * 10))  # Normalize to 0-20
            print(f"  L{i:2d} [{layer_type}]: {sim:+.3f} {bar}")


# ============================================================================
# CELL 6: DATA LOADING
# ============================================================================

def load_training_data(cfg: TrainConfig, tokenizer) -> torch.Tensor:
    """Stream and tokenize training data."""
    from datasets import load_dataset
    from tqdm import tqdm
    
    print(f"🌊 Streaming: {cfg.dataset_name}/{cfg.dataset_subset}")
    
    dataset = load_dataset(
        cfg.dataset_name, 
        name=cfg.dataset_subset, 
        split="train", 
        streaming=True
    )
    
    token_buffer = []
    pbar = tqdm(total=cfg.target_tokens, desc="Tokenizing", unit="tok")
    
    for example in dataset:
        tokens = tokenizer.encode(example['text']) + [tokenizer.eos_token_id]
        token_buffer.extend(tokens)
        pbar.update(len(tokens))
        
        if len(token_buffer) >= cfg.target_tokens:
            break
    
    pbar.close()
    
    all_tokens = torch.tensor(token_buffer[:cfg.target_tokens], dtype=torch.long)
    
    # Cleanup
    del token_buffer
    import gc; gc.collect()
    
    needed = cfg.steps * cfg.batch_size * cfg.seq_len * cfg.accum_steps
    coverage = len(all_tokens) / needed
    
    print(f"✓ Loaded {len(all_tokens):,} tokens")
    print(f"  Coverage: {coverage:.1f}x training needs")
    
    return all_tokens


def get_batch(
    all_tokens: torch.Tensor, 
    batch_size: int, 
    seq_len: int, 
    device: str
) -> tuple:
    """Sample a random batch from token buffer."""
    ix = torch.randint(len(all_tokens) - seq_len - 1, (batch_size,))
    x = torch.stack([all_tokens[i:i+seq_len] for i in ix])
    y = torch.stack([all_tokens[i+1:i+seq_len+1] for i in ix])
    return x.to(device), y.to(device)


# ============================================================================
# CELL 7: TRAINING LOOP
# ============================================================================

def train(
    model: nn.Module,
    all_tokens: torch.Tensor,
    tokenizer,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    device: str = "cuda"
) -> dict:
    """
    Main training loop with monitoring.
    
    Returns dict with training history.
    """
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.lr,
        betas=train_cfg.betas,
        weight_decay=train_cfg.weight_decay
    )
    
    # Tracking
    history = {
        'loss': [],
        'grad_norm': [],
        'lr': [],
        'niah': [],  # (step, ratio)
    }
    
    random_chance = 1.0 / tokenizer.vocab_size
    start_time = time.time()
    
    print("\n" + "=" * 60)
    print(f"TRAINING ({train_cfg.steps} steps)")
    print(f"  Effective batch: {train_cfg.effective_batch_size}")
    print(f"  Tokens/step: {train_cfg.effective_batch_size * train_cfg.seq_len:,}")
    print("=" * 60 + "\n")
    
    optimizer.zero_grad()
    
    for step in range(train_cfg.steps):
        # Learning rate schedule: linear warmup + cosine decay
        if step < train_cfg.warmup_steps:
            lr = train_cfg.lr * (step + 1) / train_cfg.warmup_steps
        else:
            progress = (step - train_cfg.warmup_steps) / (train_cfg.steps - train_cfg.warmup_steps)
            lr = train_cfg.lr * 0.5 * (1 + math.cos(math.pi * progress))
        
        for pg in optimizer.param_groups:
            pg['lr'] = lr
        
        # Gradient accumulation
        accum_loss = 0.0
        
        for _ in range(train_cfg.accum_steps):
            x, y = get_batch(all_tokens, train_cfg.batch_size, train_cfg.seq_len, device)
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, loss, _ = model(x, y)  # 3-tuple: logits, loss, states
                loss_scaled = loss / train_cfg.accum_steps
            
            loss_scaled.backward()
            accum_loss += loss.item()
        
        # Gradient clipping and step
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        
        # Record
        avg_loss = accum_loss / train_cfg.accum_steps
        history['loss'].append(avg_loss)
        history['grad_norm'].append(total_norm.item())
        history['lr'].append(lr)
        
        # Logging
        if step % train_cfg.log_interval == 0:
            avg_recent = np.mean(history['loss'][-50:]) if len(history['loss']) >= 50 else np.mean(history['loss'])
            elapsed = time.time() - start_time
            tps = (step + 1) * train_cfg.effective_batch_size * train_cfg.seq_len / elapsed
            
            print(f"[{step:5d}/{train_cfg.steps}] loss={avg_recent:.4f} | "
                  f"grad={total_norm.item():.3f} | lr={lr:.2e} | {tps:,.0f} tok/s")
        
        # Gradient health check
        if (step + 1) % train_cfg.grad_log_interval == 0:
            print_gradient_summary(model)
        
        # NIAH checkpoints
        if (step + 1) in train_cfg.niah_checkpoints:
            niah = needle_test(model, tokenizer, train_cfg.seq_len, n_trials=30, device=device)
            ratio = niah['ratio']
            history['niah'].append((step + 1, ratio))
            
            status = "🟢" if ratio > 1.0 else "🟡" if ratio > 0.5 else "🔴"
            print(f"  >>> {status} NIAH@{step+1}: {ratio:.2f}x random")
            
            model.train()
    
    # Final stats
    elapsed = time.time() - start_time
    final_tps = train_cfg.steps * train_cfg.effective_batch_size * train_cfg.seq_len / elapsed
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Time: {elapsed/60:.1f} min")
    print(f"  Speed: {final_tps:,.0f} tok/s")
    print(f"  Initial loss: {np.mean(history['loss'][:50]):.4f}")
    print(f"  Final loss: {np.mean(history['loss'][-50:]):.4f}")
    
    return history


# ============================================================================
# CELL 8: EVALUATION
# ============================================================================

def evaluate(model: nn.Module, tokenizer, device: str = "cuda"):
    """Run full evaluation suite."""
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    
    random_chance = 1.0 / tokenizer.vocab_size
    
    # NIAH at multiple sequence lengths
    print("\n🎯 Needle-in-a-Haystack:")
    print("-" * 50)
    
    results = {}
    for length in [128, 256, 512, 1024]:
        niah = needle_test(model, tokenizer, length, n_trials=50, device=device)
        results[length] = niah
        
        status = "🟢" if niah['ratio'] > 1.0 else "🟡" if niah['ratio'] > 0.5 else "🔴"
        print(f"  {status} NIAH@{length}: {niah['ratio']:.2f}x random "
              f"(P={niah['mean']:.2e} ± {niah['std']:.2e})")
    
    # Layer representation probe
    probe_layer_representations(model, tokenizer, device=device)
    
    return results


# ============================================================================
# CELL 9: MAIN EXECUTION
# ============================================================================

def main():
    """Main execution flow."""
    global MODEL_CFG, TRAIN_CFG
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    MODEL_CFG.vocab_size = tokenizer.vocab_size
    
    # Load data
    print("\nLoading data...")
    all_tokens = load_training_data(TRAIN_CFG, tokenizer)
    
    # Build model
    print("\nBuilding model...")
    model = GroundThinkLM(MODEL_CFG).to(DEVICE).to(torch.bfloat16)
    
    params = model.count_parameters()
    print(f"✓ Parameters: {params['total']/1e6:.2f}M")
    print(f"  - Embedding: {params['embed']/1e6:.2f}M")
    print(f"  - GDN layers: {params['gdn_layers']/1e6:.2f}M")
    print(f"  - SWA layers: {params['swa_layers']/1e6:.2f}M")
    print(f"  - FFN layers: {params['ffn_layers']/1e6:.2f}M")
    
    # Test forward/backward
    print("\nTesting forward/backward...")
    x, y = get_batch(all_tokens, TRAIN_CFG.batch_size, TRAIN_CFG.seq_len, DEVICE)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits, loss, _ = model(x, y)  # 3-tuple return
    loss.backward()
    print(f"✓ Forward: loss={loss.item():.4f}")
    print(f"✓ Backward: OK")
    print(f"✓ Peak memory: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
    model.zero_grad()
    
    # Train
    history = train(model, all_tokens, tokenizer, MODEL_CFG, TRAIN_CFG, DEVICE)
    
    # Evaluate
    results = evaluate(model, tokenizer, DEVICE)
    
    return model, history, results


# Entry point
if __name__ == "__main__":
    if not FLA_AVAILABLE:
        print("\n⚠ Cannot proceed without FLA library.")
        print("  Install with: pip install flash-linear-attention")
    else:
        model, history, results = main()
