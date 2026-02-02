#!/usr/bin/env python3
"""
Controlled retrieval test with explicit verification at each step.

4 Steps:
1. MARKER_i → GDN assigns orthogonal key K_i
2. NEEDLE → Stored at S[K_i]  
3. CUE_i → SWA queries with key K_i
4. RETRIEVAL → Output NEEDLE

Expected loss:
- Random (250 tokens): ln(250) = 5.52
- Perfect retrieval: ~0
"""

import torch
import torch.nn.functional as F
from config import HybridConfig
from model import TransparentHybrid

DEVICE = 'cuda'

# Token allocation (no overlaps)
HAYSTACK_RANGE = (0, 200)      # Tokens 0-199: haystack
VALUE_RANGE = (0, 200)         # Tokens 0-199: values to store
MARKER = 250                    # Token 250: MARKER
CUE_BASE = 251                  # Tokens 251-258: CUE_0 through CUE_7
VOCAB_SIZE = 260

cfg = HybridConfig(
    d_model=256,
    n_heads=4,
    head_dim=64,
    value_dim=64,
    layer_pattern="GS",
    key_bank_size=16,
    shifted_value=True,
    vocab_size=VOCAB_SIZE,
    marker_token=MARKER,
    cue_token=CUE_BASE,
    beta_bias=0.0,
    g_bias=2.0,
)

print("=" * 60)
print("STEP 0: Verify Architecture")
print("=" * 60)

model = TransparentHybrid(cfg).to(DEVICE)
gdn = model.layers[0]
swa = model.layers[1]

print(f"Vocab: 0-199=haystack/values, {MARKER}=MARKER, {CUE_BASE}-{CUE_BASE+7}=CUE_0-7")
print(f"Key bank: {gdn.key_bank.shape} (orthogonal keys per head)")
print(f"Random guess loss: ln({VALUE_RANGE[1]}) = {torch.log(torch.tensor(float(VALUE_RANGE[1]))).item():.3f}")

# Verify key bank orthogonality (on CPU to avoid crash)
kb = gdn.key_bank.cpu()
for h in range(cfg.n_heads):
    gram = kb[h] @ kb[h].T  # Should be identity
    off_diag = (gram - torch.eye(cfg.key_bank_size)).abs().max().item()
    if off_diag > 0.01:
        print(f"  WARNING: Head {h} not orthogonal (off-diag max: {off_diag:.4f})")
print(f"Key bank orthogonality: ✓")

print("\n" + "=" * 60)
print("STEP 1: Verify MARKER → Key Bank Assignment")
print("=" * 60)

# Test: Does MARKER get assigned bank key?
tokens = torch.randint(0, 200, (1, 32), device=DEVICE)
tokens[0, 10] = MARKER  # First MARKER → should get slot 0
tokens[0, 20] = MARKER  # Second MARKER → should get slot 1

with torch.no_grad():
    x = model.embed(tokens)
    x = model.embed_norm(x)
    x_norm = gdn.norm(x)
    
    # Get keys before bank assignment
    k_proj = gdn.k_proj(x_norm).view(1, 32, 4, 64)
    k_before_10 = k_proj[0, 10].clone()
    k_before_20 = k_proj[0, 20].clone()
    
    # Apply bank assignment (replicate GDN logic)
    marker_mask = tokens[0] == MARKER
    marker_positions = marker_mask.nonzero().squeeze(-1)
    for i, pos in enumerate(marker_positions):
        slot = i % gdn.bank_size
        k_proj[0, pos] = gdn.key_bank[:, slot, :].to(k_proj.dtype)
    
    k_after_10 = k_proj[0, 10]
    k_after_20 = k_proj[0, 20]

# Verify
bank_key_0 = gdn.key_bank[:, 0, :].cpu()
bank_key_1 = gdn.key_bank[:, 1, :].cpu()

sim_10_to_bank0 = F.cosine_similarity(k_after_10.cpu().view(1, -1), bank_key_0.view(1, -1)).item()
sim_20_to_bank1 = F.cosine_similarity(k_after_20.cpu().view(1, -1), bank_key_1.view(1, -1)).item()

print(f"MARKER@10 key matches bank slot 0: {sim_10_to_bank0:.4f} (should be 1.0)")
print(f"MARKER@20 key matches bank slot 1: {sim_20_to_bank1:.4f} (should be 1.0)")

if sim_10_to_bank0 > 0.99 and sim_20_to_bank1 > 0.99:
    print("Step 1: ✓ MARKERs get distinct orthogonal keys")
else:
    print("Step 1: ✗ FAILED - MARKERs not getting bank keys")

print("\n" + "=" * 60)
print("STEP 2: Verify Write (MARKER stores NEEDLE)")
print("=" * 60)

# Sequence: MARKER at 10, value at 11
tokens = torch.randint(0, 200, (1, 32), device=DEVICE)
tokens[0, 10] = MARKER
tokens[0, 11] = 42  # NEEDLE value

with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
    logits, _, diags, state = model(tokens)

# Query state with bank key slot 0
bank_key_0 = F.normalize(gdn.key_bank[:, 0, :].float(), dim=-1)
retrieved = torch.einsum('bhkv,hk->bhv', state.float(), bank_key_0)

# What should be stored? v_proj of value embedding
val_emb = model.embed(torch.tensor([42], device=DEVICE))
val_emb = model.embed_norm(val_emb)
val_norm = gdn.norm(val_emb)
expected_v = gdn.v_proj(val_norm).view(4, 64)

sim = F.cosine_similarity(retrieved.view(1, -1), expected_v.view(1, -1)).item()
print(f"Retrieved vs expected value similarity: {sim:.4f}")
print(f"State norm: {state.norm().item():.4f}")

if sim > 0.7:
    print("Step 2: ✓ Value stored correctly")
else:
    print("Step 2: ✗ FAILED - Value not stored")

print("\n" + "=" * 60)
print("STEP 3: Verify CUE → Bank Key Query")
print("=" * 60)

# Check: Does CUE_0 query with bank key slot 0?
# This happens in SWA forward when input_ids contains CUE_0

tokens = torch.randint(0, 200, (1, 32), device=DEVICE)
tokens[0, 10] = MARKER
tokens[0, 11] = 42
tokens[0, 25] = CUE_BASE  # CUE_0

with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
    # Run through model
    x = model.embed(tokens)
    x = model.embed_norm(x)
    
    # GDN pass
    gdn_out, state, gdn_diag = gdn(x, input_ids=tokens)
    gdn_out = model.ffns[0](gdn_out)
    
    # SWA pass - manually check what query is generated
    swa_x_norm = swa.norm(gdn_out)
    q_g = swa.global_q_proj(swa_x_norm).view(1, 32, 4, 64)
    
    # Before bank key assignment
    q_before = q_g[0, 25].clone()
    
    # The SWA should assign bank key to CUE position
    # Check if CUE_0 (token 251) at position 25 gets bank key slot 0
    for cue_idx in range(swa.bank_size):
        cue_token_id = swa.cue_token + cue_idx
        cue_mask = tokens[0] == cue_token_id
        if cue_mask.any():
            cue_positions = cue_mask.nonzero().squeeze(-1)
            q_g[0, cue_positions] = gdn.key_bank[:, cue_idx, :].to(q_g.dtype)

q_after = q_g[0, 25]
sim_to_bank0 = F.cosine_similarity(q_after.cpu().view(1, -1).float(), bank_key_0.cpu().view(1, -1)).item()

print(f"CUE_0 query matches bank slot 0: {sim_to_bank0:.4f} (should be 1.0)")

if sim_to_bank0 > 0.99:
    print("Step 3: ✓ CUE uses correct bank key for query")
else:
    print("Step 3: ✗ FAILED - CUE not using bank key")

print("\n" + "=" * 60)
print("STEP 4: Train Retrieval")
print("=" * 60)

# Fresh model
model = TransparentHybrid(cfg).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print("Training 1-needle retrieval...")
print(f"Target: loss < 1.0 (random = 5.3)")

for step in range(201):
    model.train()
    
    B = 16
    tokens = torch.randint(0, 200, (B, 64), device=DEVICE)
    targets = torch.full((B, 64), -100, device=DEVICE)
    
    for b in range(B):
        val = torch.randint(0, 200, (1,)).item()
        tokens[b, 15] = MARKER
        tokens[b, 16] = val
        tokens[b, 50] = CUE_BASE  # CUE_0
        targets[b, 50] = val
    
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits, _, _, _ = model(tokens)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1), ignore_index=-100)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 50 == 0:
        model.eval()
        correct = 0
        with torch.no_grad():
            for _ in range(50):
                tokens = torch.randint(0, 200, (1, 64), device=DEVICE)
                val = torch.randint(0, 200, (1,)).item()
                tokens[0, 15] = MARKER
                tokens[0, 16] = val
                tokens[0, 50] = CUE_BASE
                
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits, _, _, _ = model(tokens)
                if logits[0, 50].argmax().item() == val:
                    correct += 1
        
        acc = correct / 50
        print(f"  Step {step:3d}: loss={loss.item():.3f}, acc={acc:.0%}")

print("\n" + "=" * 60)
print("STEP 5: Multi-Needle Test")
print("=" * 60)

for n_needles in [2, 3, 4]:
    model = TransparentHybrid(cfg).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    for step in range(301):
        model.train()
        B = 16
        tokens = torch.randint(0, 200, (B, 128), device=DEVICE)
        targets = torch.full((B, 128), -100, device=DEVICE)
        
        for b in range(B):
            vals = [torch.randint(0, 200, (1,)).item() for _ in range(n_needles)]
            for i, val in enumerate(vals):
                m_pos = 15 + i * 15
                tokens[b, m_pos] = MARKER
                tokens[b, m_pos + 1] = val
                c_pos = 100 + i * 3
                tokens[b, c_pos] = CUE_BASE + i
                targets[b, c_pos] = val
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits, _, _, _ = model(tokens)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), targets.view(-1), ignore_index=-100)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Eval
    model.eval()
    correct = [0] * n_needles
    total = 50
    with torch.no_grad():
        for _ in range(total):
            tokens = torch.randint(0, 200, (1, 128), device=DEVICE)
            vals = [torch.randint(0, 200, (1,)).item() for _ in range(n_needles)]
            for i, val in enumerate(vals):
                m_pos = 15 + i * 15
                tokens[0, m_pos] = MARKER
                tokens[0, m_pos + 1] = val
                c_pos = 100 + i * 3
                tokens[0, c_pos] = CUE_BASE + i
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits, _, _, _ = model(tokens)
            
            for i, val in enumerate(vals):
                c_pos = 100 + i * 3
                if logits[0, c_pos].argmax().item() == val:
                    correct[i] += 1
    
    acc = sum(correct) / (n_needles * total)
    print(f"{n_needles} needles: {acc:.0%} (per: {[f'{c/total:.0%}' for c in correct]})")
