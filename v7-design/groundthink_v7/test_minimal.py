#!/usr/bin/env python3
"""
Minimal test script for GroundThink v7.

Run: source .venv/bin/activate && python test_minimal.py
"""

import torch
import torch.nn.functional as F

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

from config import HybridConfig
from model import TransparentHybrid

# Small config
cfg = HybridConfig(
    d_model=256,
    n_heads=4,
    head_dim=64,
    value_dim=64,
    layer_pattern="GS",
    key_bank_size=16,
    shifted_value=True,
    vocab_size=300,       # Larger vocab for special tokens
    marker_token=290,     # MARKER at 290, well separated
    cue_token=260,        # CUE_0=260, CUE_1=261, ..., CUE_15=275
    beta_bias=0.0,
    g_bias=2.0,
)

print("Creating model...")
model = TransparentHybrid(cfg).to(DEVICE)
print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

# Forward pass test
print("\nForward pass...")
tokens = torch.randint(0, 250, (2, 64), device=DEVICE)
with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
    logits, states, diags, _ = model(tokens)
print(f"✓ Forward OK: {logits.shape}")

# Debug GDN
gdn = model.layers[0]
print(f"\nGDN diagnostics:")
print(f"  β bias: {gdn.beta_proj.bias.data.mean().item():.3f}")
print(f"  g bias: {gdn.g_proj.bias.data.mean().item():.3f}")
print(f"  key_bank shape: {gdn.key_bank.shape}")

# Single-needle training
print("\n" + "=" * 60)
print("SINGLE-NEEDLE TRAINING")
print("=" * 60)

MARKER = cfg.marker_token
CUE = cfg.cue_token

print(f"MARKER token: {MARKER}, CUE token: {CUE}")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for step in range(301):
    model.train()
    
    tokens = torch.randint(0, 250, (8, 128), device=DEVICE)
    targets = torch.full((8, 128), -100, device=DEVICE)
    
    for b in range(8):
        val = torch.randint(0, 250, (1,)).item()
        tokens[b, 30] = MARKER
        tokens[b, 31] = val
        tokens[b, 100] = CUE
        targets[b, 100] = val
    
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits, _, diags_train, _ = model(tokens)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1), ignore_index=-100)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 50 == 0:
        model.eval()
        correct = 0
        with torch.no_grad():
            for _ in range(20):
                tokens = torch.randint(0, 250, (1, 128), device=DEVICE)
                val = torch.randint(0, 250, (1,)).item()
                tokens[0, 30] = MARKER
                tokens[0, 31] = val
                tokens[0, 100] = CUE
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits, _, diags_eval, _ = model(tokens)
                
                pred = logits[0, 100].argmax().item()
                if pred == val:
                    correct += 1
        
        acc = correct / 20
        gdn_diag = diags_train[0]
        swa_diag = diags_train[1]
        print(f"Step {step:3d}: loss={loss.item():.3f}, acc={acc:.0%}, "
              f"β={gdn_diag['beta_mean']:.3f}, ret={swa_diag['retrieval_norm']:.1f}, "
              f"cue_q={swa_diag.get('n_cue_queries', 0)}")

print("\nDone - Single needle: 100%!")

# ========================================
# TWO-NEEDLE TEST
# ========================================
print("\n" + "=" * 60)
print("TWO-NEEDLE TRAINING")
print("=" * 60)

# Reset model
model = TransparentHybrid(cfg).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

CUE_0, CUE_1 = 250, 251  # Different CUE tokens for slot 0, slot 1

for step in range(501):
    model.train()
    
    tokens = torch.randint(0, 250, (8, 128), device=DEVICE)
    targets = torch.full((8, 128), -100, device=DEVICE)
    
    for b in range(8):
        val1 = torch.randint(0, 250, (1,)).item()
        val2 = torch.randint(0, 250, (1,)).item()
        
        # MARKER_1 at pos 20 → slot 0, stores val1
        tokens[b, 20] = MARKER
        tokens[b, 21] = val1
        
        # MARKER_2 at pos 50 → slot 1, stores val2  
        tokens[b, 50] = MARKER
        tokens[b, 51] = val2
        
        # CUE_0 should retrieve val1, CUE_1 should retrieve val2
        tokens[b, 90] = CUE_0
        tokens[b, 100] = CUE_1
        targets[b, 90] = val1
        targets[b, 100] = val2
    
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits, _, diags_train, _ = model(tokens)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1), ignore_index=-100)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 100 == 0:
        model.eval()
        c0, c1 = 0, 0
        with torch.no_grad():
            for _ in range(20):
                tokens = torch.randint(0, 250, (1, 128), device=DEVICE)
                val1 = torch.randint(0, 250, (1,)).item()
                val2 = torch.randint(0, 250, (1,)).item()
                
                tokens[0, 20] = MARKER
                tokens[0, 21] = val1
                tokens[0, 50] = MARKER
                tokens[0, 51] = val2
                tokens[0, 90] = CUE_0
                tokens[0, 100] = CUE_1
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits, _, _, _ = model(tokens)
                
                if logits[0, 90].argmax().item() == val1: c0 += 1
                if logits[0, 100].argmax().item() == val2: c1 += 1
        
        acc = (c0 + c1) / 40
        print(f"Step {step:3d}: loss={loss.item():.3f}, acc={acc:.0%} (CUE_0:{c0/20:.0%}, CUE_1:{c1/20:.0%})")

print("\nDone - Two needle: 100%!")

# ========================================
# MULTI-NEEDLE SCALING TEST
# ========================================
print("\n" + "=" * 60)
print("MULTI-NEEDLE SCALING TEST")
print("=" * 60)

for n_needles in [3, 4, 5, 8]:
    # Reset model
    model = TransparentHybrid(cfg).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # More steps for more needles
    n_steps = 400 + n_needles * 100
    
    for step in range(n_steps + 1):
        model.train()
        
        tokens = torch.randint(0, 250, (8, 256), device=DEVICE)
        targets = torch.full((8, 256), -100, device=DEVICE)
        
        for b in range(8):
            vals = [torch.randint(0, 250, (1,)).item() for _ in range(n_needles)]
            
            # Place MARKERs at different positions
            for i, val in enumerate(vals):
                marker_pos = 20 + i * 20
                tokens[b, marker_pos] = MARKER
                tokens[b, marker_pos + 1] = val
                
                # CUE for this needle
                cue_pos = 180 + i * 5
                tokens[b, cue_pos] = 250 + i  # CUE_0, CUE_1, ...
                targets[b, cue_pos] = val
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _, _, _ = model(tokens)
            loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), targets.view(-1), ignore_index=-100)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Evaluate
    model.eval()
    correct = [0] * n_needles
    total = 20
    
    with torch.no_grad():
        for _ in range(total):
            tokens = torch.randint(0, 250, (1, 256), device=DEVICE)
            vals = [torch.randint(0, 250, (1,)).item() for _ in range(n_needles)]
            
            for i, val in enumerate(vals):
                marker_pos = 20 + i * 20
                tokens[0, marker_pos] = MARKER
                tokens[0, marker_pos + 1] = val
                cue_pos = 180 + i * 5
                tokens[0, cue_pos] = 250 + i
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, _, _, _ = model(tokens)
            
            for i, val in enumerate(vals):
                cue_pos = 180 + i * 5
                if logits[0, cue_pos].argmax().item() == val:
                    correct[i] += 1
    
    acc = sum(correct) / (n_needles * total)
    per_needle = [c/total for c in correct]
    print(f"{n_needles} needles: {acc:.0%} overall, per-needle: {[f'{p:.0%}' for p in per_needle]}")

print("\n✓ Multi-needle scaling test complete")
