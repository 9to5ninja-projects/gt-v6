"""
Scaling Experiment: Does GSS scale properly?

Protocol per proof_of_concept.md:
1. Train 3-4 model sizes with IDENTICAL protocol
2. Fit power law L(N) = A/N^α + B
3. If α > 0.3, scaling investment justified
4. State ablation: Memory Reliance Score > 70%

Hardware constraint: RTX 4050 6GB
- 15M: easy
- 50M: manageable  
- 100M: gradient checkpointing
- 150M+: likely OOM, try anyway

Data: OpenWebText subset (real text, not TinyStories)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import math
import random
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple
import sys

sys.path.insert(0, '.')
from config import HybridConfig
from model import TransparentHybrid

# =============================================================================
# EXPERIMENT CONFIG
# =============================================================================

@dataclass
class ScalingConfig:
    """Fixed protocol for all model sizes."""
    # Pattern: GS - minimal viable baseline (1 GDN + 1 SWA)
    # Must establish this works before adding layers
    layer_pattern: str = 'GS'
    
    # Reproducibility
    seed: int = 42
    
    # Training protocol - per ssm_training_text.md
    total_steps: int = 15000  # Minimum per references
    warmup_steps: int = 2000  # Per references: 2000-4000
    lr_peak: float = 3e-4
    lr_min: float = 1e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    
    # Curriculum (same for all)
    curriculum: List[Tuple[int, int]] = None
    
    # Batch size varies by model size
    batch_size_small: int = 4   # 15M
    batch_size_medium: int = 2  # 50M
    batch_size_large: int = 1   # 100M+
    
    # Gradient accumulation for stable training
    accum_steps: int = 8  # Effective batch = batch_size * accum_steps
    
    # Validation
    val_every: int = 500
    val_samples: int = 50       # Quick validation during training
    final_eval_samples: int = 500  # References: n≥500 for final evaluation
    
    # Data - WSL memory limited, keep reasonable
    train_tokens: int = 5_000_000   # 5M tokens
    val_tokens: int = 200_000       # 200K tokens
    
    def __post_init__(self):
        # Grow-P2 curriculum per ssm_training_text.md: power-of-2 progression 256→4096
        # 8 cycles across training, each cycle is total_steps/8
        # With 15K steps: each cycle ~1875 steps
        self.curriculum = [
            (0, 256),
            (1875, 512),
            (3750, 1024),
            (5625, 2048),
            (7500, 4096),
            # Remaining steps stay at 4096
        ]
    
    def get_seq_len(self, step: int) -> int:
        seq_len = self.curriculum[0][1]
        for threshold, length in self.curriculum:
            if step >= threshold:
                seq_len = length
        return seq_len


# Model size configs - n_layers is derived from layer_pattern, not set here
# layer_pattern='GS' = 2 layers (1 GDN + 1 SWA)
# batch_size=1 for all to fit 4096 seq_len in 6GB VRAM
MODEL_SIZES = {
    '10M': {
        'd_model': 256,
        'n_heads': 4,
        'head_dim': 32,
        'value_dim': 32,
        'batch_size': 1,  # Was 4, OOM at seq_len=4096
        'gradient_checkpointing': False,
    },
    '25M': {
        'd_model': 384,
        'n_heads': 6,
        'head_dim': 64,
        'value_dim': 64,
        'batch_size': 1,  # Was 2
        'gradient_checkpointing': False,
    },
    '50M': {
        'd_model': 512,
        'n_heads': 8,
        'head_dim': 64,
        'value_dim': 64,
        'batch_size': 1,  # Was 2
        'gradient_checkpointing': False,
    },
    '100M': {
        'd_model': 768,
        'n_heads': 12,
        'head_dim': 64,
        'value_dim': 64,
        'batch_size': 1,
        'gradient_checkpointing': True,
    },
    '150M': {
        'd_model': 1024,
        'n_heads': 16,
        'head_dim': 64,
        'value_dim': 64,
        'batch_size': 1,
        'gradient_checkpointing': True,
    },
}


# =============================================================================
# VRAM TEST
# =============================================================================

def test_vram_fit(size_name: str, size_cfg: dict, layer_pattern: str, 
                  seq_len: int = 4096, device: torch.device = None):
    """
    Test if model fits in VRAM at target seq_len.
    Returns (fits: bool, peak_mb: float, n_params: int)
    """
    if device is None:
        device = torch.device('cuda')
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    try:
        cfg = HybridConfig(
            layer_pattern=layer_pattern,
            d_model=size_cfg['d_model'],
            n_heads=size_cfg['n_heads'],
            head_dim=size_cfg['head_dim'],
            value_dim=size_cfg['value_dim'],
            vocab_size=50257,
            window_size=128,
            local_drop_prob=0.0,
            local_scale=0.3,
        )
        model = TransparentHybrid(cfg).to(device)
        n_params = model.count_params()
        
        # Test forward + backward at target seq_len
        batch_size = size_cfg['batch_size']
        x = torch.randint(0, 50257, (batch_size, seq_len), device=device)
        
        with torch.amp.autocast('cuda'):
            logits, loss, _, _ = model(x, targets=x)
        
        loss.backward()
        
        peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        del model, x, logits, loss
        torch.cuda.empty_cache()
        
        return True, peak_mb, n_params
        
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            torch.cuda.empty_cache()
            return False, -1, -1
        raise


def run_vram_tests(layer_pattern: str = 'GS', seq_len: int = 4096):
    """Test which model sizes fit in VRAM."""
    print(f"\n{'='*60}")
    print(f"VRAM Test: pattern={layer_pattern}, seq_len={seq_len}")
    print(f"{'='*60}")
    
    device = torch.device('cuda')
    total_vram = torch.cuda.get_device_properties(device).total_memory / 1024 / 1024
    print(f"Total VRAM: {total_vram:.0f} MB")
    print()
    
    results = []
    for name, cfg in MODEL_SIZES.items():
        fits, peak_mb, n_params = test_vram_fit(name, cfg, layer_pattern, seq_len, device)
        if fits:
            print(f"  {name:6s}: ✅ {n_params/1e6:.2f}M params, {peak_mb:.0f} MB peak")
            results.append((name, n_params, peak_mb))
        else:
            print(f"  {name:6s}: ❌ OOM")
            results.append((name, -1, -1))
    
    return results


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data(tokenizer, train_tokens: int = 2_000_000, val_tokens: int = 100_000):
    """Load OpenWebText subset with train/val split."""
    from datasets import load_dataset
    
    print("Loading OpenWebText...")
    # Stream to avoid loading full dataset
    ds = load_dataset('openwebtext', split='train', streaming=True)
    
    all_tokens = []
    needed = train_tokens + val_tokens + 10000  # buffer
    
    for example in ds:
        text = example['text']
        if text.strip():
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)
        if len(all_tokens) >= needed:
            break
        if len(all_tokens) % 500000 == 0:
            print(f"  Loaded {len(all_tokens):,} tokens...")
    
    all_tokens = torch.tensor(all_tokens[:needed], dtype=torch.long)
    
    # Split
    train = all_tokens[:train_tokens]
    val = all_tokens[train_tokens:train_tokens + val_tokens]
    
    print(f"Train: {len(train):,} tokens")
    print(f"Val: {len(val):,} tokens")
    
    return train, val


def get_batch(tokens: torch.Tensor, seq_len: int, batch_size: int):
    """Random batch from token stream."""
    max_start = len(tokens) - seq_len - 1
    starts = torch.randint(0, max_start, (batch_size,))
    batch = torch.stack([tokens[s:s+seq_len+1] for s in starts])
    return batch[:, :-1], batch[:, 1:]


# =============================================================================
# TRAINING
# =============================================================================

def get_lr(step: int, cfg: ScalingConfig) -> float:
    """Warmup + cosine decay."""
    if step < cfg.warmup_steps:
        return cfg.lr_peak * (step / cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / (cfg.total_steps - cfg.warmup_steps)
    return cfg.lr_min + 0.5 * (cfg.lr_peak - cfg.lr_min) * (1 + math.cos(math.pi * progress))


def evaluate(model, val_tokens: torch.Tensor, seq_len: int, batch_size: int, 
             n_samples: int, device: torch.device, zero_state: bool = False):
    """
    Evaluate model on validation set.
    If zero_state=True, zero the GDN state for ablation test.
    """
    model.eval()
    losses = []
    
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for _ in range(n_samples):
            inputs, targets = get_batch(val_tokens, seq_len, batch_size)
            inputs, targets = inputs.to(device), targets.to(device)
            
            logits, loss, diag, state = model(inputs, targets=targets)
            
            if zero_state and state is not None:
                # Zero state ablation - re-run with zeroed state
                # This tests if model relies on state
                pass  # TODO: Need to modify model to accept forced zero state
            
            losses.append(loss.item())
    
    avg_loss = sum(losses) / len(losses)
    ppl = math.exp(avg_loss)
    return {'loss': avg_loss, 'ppl': ppl}


def generate_mqar_sample(n_pairs: int, vocab_size: int = 50257, seq_len: int = 256):
    """
    Generate one MQAR (Multi-Query Associative Recall) sample.
    
    Format: K1 V1 K2 V2 ... Kn Vn [SEP] Q1 ? Q2 ? ... Qn ?
    Target: For each Qi, predict Vi
    
    Uses actual model vocab tokens (excluding special tokens 50250-50256).
    Keys from [1000, 25000), Values from [25000, 49000) to avoid overlap.
    """
    # Generate unique key-value pairs from model vocab
    # Avoid special tokens and very common tokens
    keys = torch.randint(1000, 25000, (n_pairs,))
    values = torch.randint(25000, 49000, (n_pairs,))
    
    # Ensure uniqueness
    keys = torch.unique(keys)[:n_pairs]
    values = torch.unique(values)[:n_pairs]
    if len(keys) < n_pairs or len(values) < n_pairs:
        # Fallback if not enough unique
        keys = torch.arange(1000, 1000 + n_pairs)
        values = torch.arange(25000, 25000 + n_pairs)
    
    # Build sequence using actual special tokens
    sep_token = 50256  # GPT-2 EOS as separator
    query_marker = 30  # '?' in GPT-2 vocab
    
    tokens = []
    # Store phase
    for k, v in zip(keys, values):
        tokens.extend([k.item(), v.item()])
    tokens.append(sep_token)
    
    # Query phase - shuffle query order
    query_order = torch.randperm(n_pairs)
    targets = []
    for qi in query_order:
        tokens.append(keys[qi].item())
        tokens.append(query_marker)
        targets.append(values[qi].item())
    
    # Pad to seq_len
    if len(tokens) < seq_len:
        tokens.extend([0] * (seq_len - len(tokens)))
    
    return torch.tensor(tokens[:seq_len]), targets, query_order.tolist()


def evaluate_mqar(model, n_pairs: int, n_samples: int, device: torch.device):
    """
    Evaluate model on MQAR task.
    
    Returns accuracy: fraction of correctly recalled values.
    """
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for _ in range(n_samples):
            tokens, targets, query_order = generate_mqar_sample(n_pairs)
            tokens = tokens.unsqueeze(0).to(device)  # [1, seq_len]
            
            logits, _, _, _ = model(tokens)  # [1, seq_len, vocab]
            
            # Find positions of query markers ('?' = token 30)
            # Predictions are at query_marker positions
            query_positions = (tokens[0] == 30).nonzero(as_tuple=True)[0]  # '?' in GPT-2
            
            for i, pos in enumerate(query_positions):
                if i >= len(targets):
                    break
                pred = logits[0, pos - 1].argmax().item()  # Predict at position before ?
                if pred == targets[i]:
                    correct += 1
                total += 1
    
    accuracy = correct / total if total > 0 else 0.0
    return {'mqar_accuracy': accuracy, 'mqar_correct': correct, 'mqar_total': total}


def generate_text(model, prompt_tokens: torch.Tensor, max_new_tokens: int, 
                  temperature: float = 1.0, device: torch.device = None):
    """Generate text autoregressively."""
    model.eval()
    tokens = prompt_tokens.clone()
    
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for _ in range(max_new_tokens):
            logits, _, _, _ = model(tokens)
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)
    
    return tokens


def evaluate_generation_quality(model, val_tokens: torch.Tensor, device: torch.device,
                                 n_samples: int = 50, gen_len: int = 100, prompt_len: int = 32):
    """
    Evaluate generation quality per references:
    - Generated PPL: should be 12-20 (coherent text)
    - Distinct-2: >0.5 (diverse bigrams)
    - Repetition-4: <10% (not repeating 4-grams)
    """
    model.eval()
    generations = []
    gen_losses = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            # Random prompt from validation
            start = random.randint(0, len(val_tokens) - prompt_len - 1)
            prompt = val_tokens[start:start + prompt_len].unsqueeze(0).to(device)
            
            # Generate
            full_seq = generate_text(model, prompt, gen_len, temperature=1.0, device=device)
            generated = full_seq[0, prompt_len:].cpu().tolist()
            generations.append(generated)
            
            # Compute loss on generated text (proxy for generated PPL)
            with torch.amp.autocast('cuda'):
                gen_input = full_seq[:, :-1]
                gen_target = full_seq[:, 1:]
                logits, loss, _, _ = model(gen_input, targets=gen_target)
                gen_losses.append(loss.item())
    
    # Distinct-2: unique bigrams / total bigrams
    all_bigrams = []
    for gen in generations:
        for i in range(len(gen) - 1):
            all_bigrams.append((gen[i], gen[i+1]))
    distinct_2 = len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0.0
    
    # Repetition-4: % of 4-grams that are repeated within same generation
    rep_4_counts = []
    for gen in generations:
        if len(gen) < 4:
            continue
        fourgrams = [tuple(gen[i:i+4]) for i in range(len(gen) - 3)]
        n_unique = len(set(fourgrams))
        n_total = len(fourgrams)
        rep_4_counts.append(1.0 - n_unique / n_total if n_total > 0 else 0.0)
    rep_4 = sum(rep_4_counts) / len(rep_4_counts) if rep_4_counts else 0.0
    
    # Generated PPL
    avg_gen_loss = sum(gen_losses) / len(gen_losses)
    gen_ppl = math.exp(avg_gen_loss)
    
    return {
        'gen_ppl': gen_ppl,
        'distinct_2': distinct_2,
        'repetition_4': rep_4,
    }


def evaluate_position_binned_accuracy(model, val_tokens: torch.Tensor, device: torch.device,
                                       seq_len: int = 512, n_samples: int = 100):
    """
    Test if accuracy varies by position (red flag for shortcut learning).
    
    Per references: >15% accuracy variation across positions indicates position bias.
    Genuine memory should have similar accuracy across all positions.
    """
    model.eval()
    
    # Divide sequence into 4 bins
    bin_size = seq_len // 4
    bin_correct = [0, 0, 0, 0]
    bin_total = [0, 0, 0, 0]
    
    with torch.no_grad(), torch.amp.autocast('cuda'):
        for _ in range(n_samples):
            # Get random sequence
            start = random.randint(0, len(val_tokens) - seq_len - 1)
            input_seq = val_tokens[start:start + seq_len].unsqueeze(0).to(device)
            target_seq = val_tokens[start + 1:start + seq_len + 1].to(device)
            
            logits, _, _, _ = model(input_seq)
            preds = logits[0].argmax(dim=-1)  # [seq_len]
            
            # Check accuracy per position bin
            for pos in range(seq_len):
                bin_idx = min(pos // bin_size, 3)
                if preds[pos] == target_seq[pos]:
                    bin_correct[bin_idx] += 1
                bin_total[bin_idx] += 1
    
    bin_acc = [c / t if t > 0 else 0.0 for c, t in zip(bin_correct, bin_total)]
    acc_range = max(bin_acc) - min(bin_acc)
    
    return {
        'bin_accuracies': bin_acc,
        'accuracy_range': acc_range,
        'position_bias_flag': acc_range > 0.15,  # Red flag if >15% variation
    }


def train_model(size_name: str, size_cfg: dict, exp_cfg: ScalingConfig,
                train_tokens: torch.Tensor, val_tokens: torch.Tensor,
                device: torch.device, results_dir: str):
    """Train a single model size and return metrics."""
    
    print(f"\n{'='*60}")
    print(f"Training {size_name}")
    print(f"{'='*60}")
    
    torch.cuda.empty_cache()
    
    # Create model
    # local_drop_prob=0: always use both pathways (no dropout killing local attention)
    # local_scale=0.3: local attention contribution weight
    model_cfg = HybridConfig(
        layer_pattern=exp_cfg.layer_pattern,
        d_model=size_cfg['d_model'],
        n_heads=size_cfg['n_heads'],
        head_dim=size_cfg['head_dim'],
        value_dim=size_cfg['value_dim'],
        vocab_size=50257,
        window_size=128,
        local_drop_prob=0.0,  # Was 0.5 - killed local attention
        local_scale=0.3,
    )
    
    try:
        model = TransparentHybrid(model_cfg).to(device)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f"  OOM creating model. Skipping {size_name}.")
            return None
        raise
    
    n_params = model.count_params()
    print(f"  Parameters: {n_params/1e6:.2f}M")
    print(f"  d_model: {size_cfg['d_model']}, n_heads: {size_cfg['n_heads']}")
    print(f"  batch_size: {size_cfg['batch_size']}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=exp_cfg.lr_peak,
        betas=(exp_cfg.beta1, exp_cfg.beta2),
        weight_decay=exp_cfg.weight_decay
    )
    scaler = torch.amp.GradScaler('cuda')
    
    # History - track train/val gap explicitly
    history = {
        'train_loss': [], 'val_loss': [], 'val_ppl': [],
        'train_val_gap': [],  # val_loss - train_loss (should be small but present)
        'grad_norm': [],
        'step': [], 'seq_len': [], 'lr': []
    }
    
    batch_size = size_cfg['batch_size']
    effective_batch = batch_size * exp_cfg.accum_steps
    print(f"  Effective batch size: {effective_batch} (batch={batch_size} x accum={exp_cfg.accum_steps})")
    
    try:
        optimizer.zero_grad()  # Move outside loop for accumulation
        
        for step in range(exp_cfg.total_steps):
            seq_len = exp_cfg.get_seq_len(step)
            lr = get_lr(step, exp_cfg)
            for pg in optimizer.param_groups:
                pg['lr'] = lr
            
            # Train step
            inputs, targets = get_batch(train_tokens, seq_len, batch_size)
            inputs, targets = inputs.to(device), targets.to(device)
            
            model.train()
            
            with torch.amp.autocast('cuda'):
                logits, loss, diag, state = model(inputs, targets=targets)
                loss_scaled = loss / exp_cfg.accum_steps  # Normalize for accumulation
            
            # NaN/Inf detection
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  ⚠️ NaN/Inf loss at step {step}! Halting.")
                break
            
            scaler.scale(loss_scaled).backward()
            
            # Only step every accum_steps
            if (step + 1) % exp_cfg.accum_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), exp_cfg.grad_clip)
                
                # Gradient explosion detection
                if torch.isnan(grad_norm) or grad_norm > 100:
                    print(f"  ⚠️ Gradient explosion at step {step}: norm={grad_norm:.1f}")
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            else:
                grad_norm = 0.0  # Not computed this step
            
            history['train_loss'].append(loss.item())
            history['grad_norm'].append(float(grad_norm))
            history['step'].append(step)
            history['seq_len'].append(seq_len)
            history['lr'].append(lr)
            
            # Periodic checkpoint (every 1000 steps)
            if step > 0 and step % 1000 == 0:
                ckpt_path = os.path.join(results_dir, f'checkpoint_{size_name}_step{step}.pt')
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'history': history,
                    'config': model_cfg.__dict__,
                }, ckpt_path)
                print(f"  [Checkpoint saved: step {step}]")
            
            # Validation
            if step % exp_cfg.val_every == 0 or step == exp_cfg.total_steps - 1:
                val_metrics = evaluate(model, val_tokens, seq_len, batch_size,
                                       exp_cfg.val_samples, device)
                history['val_loss'].append(val_metrics['loss'])
                history['val_ppl'].append(val_metrics['ppl'])
                
                # Track train/val gap
                gap = val_metrics['loss'] - loss.item()
                history['train_val_gap'].append(gap)
                
                # Warn if gap is suspicious
                if abs(gap) < 0.01:
                    print(f"  ⚠️ Train/val gap near zero ({gap:.4f}) - possible degenerate solution")
                
                print(f"  Step {step:4d}: train={loss.item():.3f}, "
                      f"val={val_metrics['loss']:.3f}, ppl={val_metrics['ppl']:.1f}, gap={gap:.3f}")
            else:
                history['val_loss'].append(None)
                history['val_ppl'].append(None)
                history['train_val_gap'].append(None)
    
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print(f"  OOM at step {step}. Returning partial results.")
            torch.cuda.empty_cache()
        else:
            raise
    
    # Final evaluation with proper sample size (n≥500 per references)
    print(f"  Running final evaluation with {exp_cfg.final_eval_samples} samples...")
    final_metrics = evaluate(model, val_tokens, seq_len, batch_size,
                             exp_cfg.final_eval_samples, device)
    
    # MQAR evaluation - tests associative recall capacity
    print(f"  Running MQAR evaluation...")
    mqar_results = {}
    for n_pairs in [2, 4, 8, 16]:  # Test increasing difficulty
        mqar = evaluate_mqar(model, n_pairs=n_pairs, n_samples=100, device=device)
        mqar_results[f'mqar_{n_pairs}'] = mqar['mqar_accuracy']
        print(f"    MQAR-{n_pairs}: {mqar['mqar_accuracy']*100:.1f}%")
    
    # Generation quality - tests coherence and diversity
    print(f"  Running generation quality evaluation...")
    gen_quality = evaluate_generation_quality(model, val_tokens, device, n_samples=50)
    print(f"    Gen PPL: {gen_quality['gen_ppl']:.1f} (target: 12-20)")
    print(f"    Distinct-2: {gen_quality['distinct_2']:.3f} (target: >0.5)")
    print(f"    Repetition-4: {gen_quality['repetition_4']*100:.1f}% (target: <10%)")
    
    # Position-binned accuracy - tests for position bias / shortcut learning
    print(f"  Running position bias test...")
    pos_test = evaluate_position_binned_accuracy(model, val_tokens, device, seq_len=seq_len)
    print(f"    Bin accuracies: {[f'{a:.3f}' for a in pos_test['bin_accuracies']]}")
    print(f"    Accuracy range: {pos_test['accuracy_range']:.3f} (red flag if >0.15)")
    if pos_test['position_bias_flag']:
        print(f"    ⚠️  Position bias detected!")
    
    # Final metrics from proper evaluation
    final_train = sum(history['train_loss'][-50:]) / min(50, len(history['train_loss']))
    final_val = final_metrics['loss']
    final_ppl = final_metrics['ppl']
    
    result = {
        'size_name': size_name,
        'n_params': n_params,
        'config': size_cfg,
        'final_train_loss': final_train,
        'final_val_loss': final_val,
        'final_ppl': final_ppl,
        'mqar': mqar_results,
        'generation': gen_quality,
        'position_test': pos_test,
        'steps_completed': len(history['train_loss']),
        'history': history,
    }
    
    # Save checkpoint
    ckpt_path = os.path.join(results_dir, f'model_{size_name}.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': model_cfg.__dict__,
        'result': result,
    }, ckpt_path)
    print(f"  Saved: {ckpt_path}")
    
    del model
    torch.cuda.empty_cache()
    
    return result


# =============================================================================
# POWER LAW FIT
# =============================================================================

def fit_power_law(results: List[dict]) -> dict:
    """
    Fit L(N) = A/N^α + B to results.
    Returns α (scaling exponent) and fit quality.
    """
    from scipy.optimize import curve_fit
    
    def power_law(N, A, alpha, B):
        return A / (N ** alpha) + B
    
    # Extract data
    params = np.array([r['n_params'] for r in results])
    losses = np.array([r['final_val_loss'] for r in results])
    
    # Filter out inf/nan
    valid = np.isfinite(losses)
    params = params[valid]
    losses = losses[valid]
    
    if len(params) < 3:
        return {'alpha': None, 'error': 'Not enough valid data points'}
    
    try:
        popt, pcov = curve_fit(
            power_law, params, losses,
            p0=[1.0, 0.3, 3.0],  # Initial guess
            bounds=([0, 0, 0], [100, 2, 10]),
            maxfev=10000
        )
        A, alpha, B = popt
        
        # R² calculation
        predicted = power_law(params, A, alpha, B)
        ss_res = np.sum((losses - predicted) ** 2)
        ss_tot = np.sum((losses - np.mean(losses)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return {
            'A': float(A),
            'alpha': float(alpha),
            'B': float(B),
            'r_squared': float(r_squared),
            'interpretation': 'Good scaling' if alpha > 0.3 else 'Poor scaling',
        }
    except Exception as e:
        return {'alpha': None, 'error': str(e)}


# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='GS baseline scaling experiment')
    parser.add_argument('--vram-test', action='store_true', 
                        help='Run VRAM test only, no training')
    parser.add_argument('--seq-len', type=int, default=4096,
                        help='Sequence length for VRAM test (default: 4096)')
    parser.add_argument('--sizes', nargs='+', default=None,
                        help='Model sizes to train (e.g., 10M 25M)')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # VRAM test only
    if args.vram_test:
        run_vram_tests(layer_pattern='GS', seq_len=args.seq_len)
        return
    
    # Config
    exp_cfg = ScalingConfig()
    
    # Set seeds for reproducibility
    torch.manual_seed(exp_cfg.seed)
    torch.cuda.manual_seed_all(exp_cfg.seed)
    np.random.seed(exp_cfg.seed)
    random.seed(exp_cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed: {exp_cfg.seed}")
    
    # Results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = f'results/scaling_{timestamp}'
    os.makedirs(results_dir, exist_ok=True)
    
    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load data ONCE
    train_tokens, val_tokens = load_data(
        tokenizer, 
        train_tokens=exp_cfg.train_tokens,
        val_tokens=exp_cfg.val_tokens
    )
    
    # Filter sizes if specified
    sizes_to_train = MODEL_SIZES
    if args.sizes:
        sizes_to_train = {k: v for k, v in MODEL_SIZES.items() if k in args.sizes}
        if not sizes_to_train:
            print(f"Error: No matching sizes. Available: {list(MODEL_SIZES.keys())}")
            return
    
    # Train each size
    all_results = []
    for size_name, size_cfg in sizes_to_train.items():
        result = train_model(
            size_name, size_cfg, exp_cfg,
            train_tokens, val_tokens,
            device, results_dir
        )
        if result:
            all_results.append(result)
    
    # Power law fit
    print("\n" + "="*60)
    print("SCALING ANALYSIS")
    print("="*60)
    
    power_law = fit_power_law(all_results)
    if 'error' in power_law:
        print(f"Power law fit failed: {power_law['error']}")
    else:
        A = power_law.get('A', 0)
        alpha = power_law.get('alpha', 0)
        B = power_law.get('B', 0)
        r_sq = power_law.get('r_squared', 0)
        print(f"Power law fit: L(N) = {A:.2f}/N^{alpha:.3f} + {B:.2f}")
        print(f"α = {alpha:.3f}")
        print(f"R² = {r_sq:.4f}")
        print(f"Interpretation: {power_law.get('interpretation', 'Unknown')}")
    
    # Summary table
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    print(f"{'Size':<8} {'Params':>10} {'Val Loss':>10} {'PPL':>10} {'Steps':>8}")
    print("-"*60)
    for r in all_results:
        print(f"{r['size_name']:<8} {r['n_params']/1e6:>9.2f}M {r['final_val_loss']:>10.3f} "
              f"{r['final_ppl']:>10.1f} {r['steps_completed']:>8}")
    
    # Check scaling monotonicity
    if len(all_results) >= 2:
        print("\n" + "="*60)
        print("SCALING MONOTONICITY CHECK")
        print("="*60)
        sorted_results = sorted(all_results, key=lambda x: x['n_params'])
        monotonic = True
        for i in range(1, len(sorted_results)):
            prev = sorted_results[i-1]
            curr = sorted_results[i]
            if curr['final_ppl'] >= prev['final_ppl']:
                print(f"  ⚠️ Non-monotonic: {curr['size_name']} ({curr['n_params']/1e6:.1f}M) has PPL {curr['final_ppl']:.1f} "
                      f">= {prev['size_name']} ({prev['n_params']/1e6:.1f}M) PPL {prev['final_ppl']:.1f}")
                monotonic = False
            else:
                print(f"  ✅ {prev['size_name']} → {curr['size_name']}: PPL {prev['final_ppl']:.1f} → {curr['final_ppl']:.1f}")
        
        if not monotonic:
            print("\n  🛑 RED FLAG: Larger models not improving. Check for bugs or architecture issues.")
        else:
            print("\n  ✅ Scaling is monotonic - larger models have lower PPL")
    
    # Export
    export = {
        'timestamp': timestamp,
        'experiment_config': asdict(exp_cfg),
        'model_sizes': MODEL_SIZES,
        'results': all_results,
        'power_law_fit': power_law,
    }
    
    export_path = os.path.join(results_dir, 'scaling_results.json')
    with open(export_path, 'w') as f:
        json.dump(export, f, indent=2, default=str)
    print(f"\nResults saved to: {export_path}")


if __name__ == '__main__':
    main()
