"""
GroundThink v7 Analysis & Training Utilities

- NIAH (Needle-In-A-Haystack) tests
- Delta Rule validation suite
- Training with curriculum learning
- Gradient analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Dict, List, Tuple
import time

__all__ = [
    'proper_niah_test', 'test_niah_by_distance', 'run_full_diagnostic',
    'multi_needle_test', 'test_state_density', 'test_state_vs_swa',
    'diagnose_state_mechanism',
    'validate_delta_rule', 'train_curriculum', 'analyze_gradients',
    'train_with_key_reg', 'compute_key_orthogonality_loss', 'compute_beta_sparsity_loss',
    'TextDataset', 'load_wikitext',
    # Memory Gym
    'MemoryGymDataset', 'create_memory_gym_loader', 
    'train_memory_gym', 'train_mixed_curriculum',
]


# =============================================================================
# NIAH TESTING
# =============================================================================

def proper_niah_test(model, seq_len=128, needle_pos=32, n_trials=30):
    """
    Needle-In-A-Haystack test with MARKER + CUE tokens.
    
    Tests the model's ability to:
        1. Store information at MARKER position
        2. Retrieve it when CUE is seen
    """
    model.eval()
    device = next(model.parameters()).device
    cfg = model.cfg
    
    correct = 0
    for _ in range(n_trials):
        needle_id = cfg.vocab_size - 3
        seq = torch.randint(0, cfg.vocab_size - 100, (1, seq_len), device=device)
        
        seq[0, needle_pos] = cfg.marker_token
        seq[0, needle_pos + 1] = needle_id
        seq[0, -1] = cfg.cue_token
        
        with torch.no_grad():
            logits, _, _, _ = model(seq)
        
        pred = logits[0, -1].argmax().item()
        if pred == needle_id:
            correct += 1
    
    acc = correct / n_trials
    print(f"  Accuracy: {acc*100:.1f}% ({correct}/{n_trials})")
    return {'accuracy': acc, 'correct': correct, 'total': n_trials}


def test_niah_by_distance(model, distances=[5, 10, 20, 40, 60, 95], n_trials=20, seq_len=128):
    """Test retrieval across varying distances."""
    print(f"\nNIAH by Distance (seq_len={seq_len}):")
    results = {}
    
    for dist in distances:
        needle_pos = max(2, seq_len - dist - 2)
        print(f"  Distance {dist:3d} (pos={needle_pos:3d}): ", end="")
        result = proper_niah_test(model, seq_len=seq_len, needle_pos=needle_pos, n_trials=n_trials)
        results[dist] = result
    
    return results


# =============================================================================
# MULTI-NEEDLE TESTING - State Density Analysis
# =============================================================================

def multi_needle_test(model, seq_len=1024, n_needles=3, n_trials=30, verbose=True):
    """
    Multi-Needle-In-A-Haystack test.
    
    Tests STATE DENSITY: Can the GDN distinguish between multiple stored facts?
    
    Setup:
        - Place N needles at different positions with different MARKER tokens
        - Each needle has a unique ID (color=X, number=Y, name=Z)
        - Query asks for ONE specific needle
        - Model must retrieve the correct one, not just "any" stored value
    
    This tests if the state is REASONING (selective retrieval) vs INDEXING (one slot).
    """
    model.eval()
    device = next(model.parameters()).device
    cfg = model.cfg
    
    # Reserve token ranges for needle types
    # We'll use marker_token, marker_token+1, marker_token+2 as different "type" markers
    # And reserve unique value ranges for each
    base_marker = cfg.marker_token
    base_value = cfg.vocab_size - 100  # Reserve top 100 tokens for needle values
    
    results = {
        'per_needle': [],
        'total_correct': 0,
        'total_trials': 0,
        'confusion_matrix': torch.zeros(n_needles, n_needles, dtype=torch.long),
    }
    
    for trial in range(n_trials):
        # Distribute needles evenly across sequence
        positions = [int((i + 1) * seq_len / (n_needles + 2)) for i in range(n_needles)]
        
        # Create haystack
        seq = torch.randint(0, cfg.vocab_size - 200, (1, seq_len), device=device)
        
        # Place needles: [MARKER_TYPE_i, VALUE_i]
        needle_values = []
        for i, pos in enumerate(positions):
            marker_type = base_marker + i  # Different marker per needle type
            value = base_value + i * 10 + (trial % 10)  # Unique value
            needle_values.append(value)
            
            seq[0, pos] = marker_type
            seq[0, pos + 1] = value
        
        # Query for a random needle type
        query_idx = trial % n_needles
        query_marker = base_marker + query_idx
        expected_value = needle_values[query_idx]
        
        # Place CUE token that indicates which needle type we want
        # Format: [CUE, QUERY_MARKER] at end
        seq[0, -2] = cfg.cue_token
        seq[0, -1] = query_marker  # "Which value was after marker type X?"
        
        with torch.no_grad():
            logits, _, _, _ = model(seq)
        
        # Predict at position -1 (after seeing query marker)
        pred = logits[0, -1].argmax().item()
        
        # Check if prediction matches expected needle
        correct = (pred == expected_value)
        results['total_correct'] += int(correct)
        results['total_trials'] += 1
        
        # Track confusion: which needle did it retrieve?
        for j, val in enumerate(needle_values):
            if pred == val:
                results['confusion_matrix'][query_idx, j] += 1
                break
    
    # Compute per-needle accuracy
    for i in range(n_needles):
        queries_for_i = (results['confusion_matrix'][i].sum().item())
        correct_for_i = results['confusion_matrix'][i, i].item()
        acc = correct_for_i / queries_for_i if queries_for_i > 0 else 0
        results['per_needle'].append({
            'needle_idx': i,
            'accuracy': acc,
            'correct': correct_for_i,
            'total': queries_for_i,
        })
    
    overall_acc = results['total_correct'] / results['total_trials']
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"MULTI-NEEDLE TEST: {n_needles} needles in {seq_len} tokens")
        print(f"{'='*60}")
        print(f"\nOverall Accuracy: {overall_acc*100:.1f}% ({results['total_correct']}/{results['total_trials']})")
        print(f"\nPer-Needle Breakdown:")
        for r in results['per_needle']:
            print(f"  Needle {r['needle_idx']}: {r['accuracy']*100:.1f}% ({r['correct']}/{r['total']})")
        
        print(f"\nConfusion Matrix (rows=query, cols=retrieved):")
        print(f"         ", end="")
        for j in range(n_needles):
            print(f"N{j:2d} ", end="")
        print()
        for i in range(n_needles):
            print(f"  Query{i}: ", end="")
            for j in range(n_needles):
                print(f"{results['confusion_matrix'][i,j]:3d} ", end="")
            print()
        
        # Diagnosis
        if overall_acc > 0.9:
            print(f"\n✓ PASS: Model distinguishes between {n_needles} stored facts!")
        elif overall_acc > 1/n_needles + 0.1:
            print(f"\n⚠ PARTIAL: Better than random ({1/n_needles*100:.0f}%), but not selective")
        else:
            print(f"\n✗ FAIL: Model has single-slot memory (can't distinguish needles)")
    
    results['overall_accuracy'] = overall_acc
    return results


def test_state_vs_swa(model, seq_len=1024, n_trials=30, verbose=True):
    """
    Verify whether retrieval uses GDN STATE or SWA (sliding window attention).
    
    Key insight: SWA has limited window_size. If needle is OUTSIDE that window
    from the query position, only GDN state can retrieve it.
    
    Test design:
        - Place needle at position P (early in sequence)
        - Query at position Q (end of sequence)  
        - If Q - P > window_size: SWA CANNOT see needle
        - Correct answer proves GDN state is being used
    
    Returns breakdown of:
        - "inside_window": needle within SWA range (either could work)
        - "outside_window": needle OUTSIDE SWA range (must be state)
    """
    model.eval()
    device = next(model.parameters()).device
    cfg = model.cfg
    window_size = getattr(cfg, 'window_size', 64)
    
    results = {
        'inside_window': {'correct': 0, 'total': 0},
        'outside_window': {'correct': 0, 'total': 0},
    }
    
    for trial in range(n_trials):
        # Test both inside and outside SWA window
        for test_type in ['inside', 'outside']:
            seq = torch.randint(0, cfg.vocab_size - 200, (1, seq_len), device=device)
            
            # Query position is always near end
            query_pos = seq_len - 2
            
            if test_type == 'inside':
                # Needle within SWA window (easy - SWA can see it)
                needle_pos = query_pos - window_size // 2  # Half window back
            else:
                # Needle OUTSIDE SWA window (hard - only state can retrieve)
                needle_pos = min(query_pos - window_size - 100, seq_len // 4)  # Well outside
            
            # Ensure valid position
            needle_pos = max(5, min(needle_pos, seq_len - 10))
            
            # Place needle: [MARKER, VALUE]
            marker = cfg.marker_token
            value = cfg.vocab_size - 50 + (trial % 50)
            seq[0, needle_pos] = marker
            seq[0, needle_pos + 1] = value
            
            # Place query: [CUE, MARKER]
            seq[0, query_pos] = cfg.cue_token
            seq[0, query_pos + 1] = marker
            
            with torch.no_grad():
                logits, _, _, _ = model(seq)
            
            pred = logits[0, query_pos + 1].argmax().item()
            correct = (pred == value)
            
            distance = query_pos - needle_pos
            key = 'inside_window' if distance <= window_size else 'outside_window'
            results[key]['correct'] += int(correct)
            results[key]['total'] += 1
    
    # Calculate accuracies
    inside_acc = results['inside_window']['correct'] / max(results['inside_window']['total'], 1)
    outside_acc = results['outside_window']['correct'] / max(results['outside_window']['total'], 1)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"STATE vs SWA VERIFICATION (window_size={window_size})")
        print(f"{'='*60}")
        print(f"\nInside SWA window (≤{window_size} tokens):")
        print(f"  Accuracy: {inside_acc*100:.1f}% ({results['inside_window']['correct']}/{results['inside_window']['total']})")
        print(f"  → Could be SWA OR state")
        print(f"\nOutside SWA window (>{window_size} tokens):")
        print(f"  Accuracy: {outside_acc*100:.1f}% ({results['outside_window']['correct']}/{results['outside_window']['total']})")
        print(f"  → MUST be state (SWA can't see)")
        
        if outside_acc > 0.8:
            print(f"\n✓ CONFIRMED: Model uses GDN STATE for retrieval!")
        elif outside_acc > 0.1:
            print(f"\n⚠ PARTIAL: Some state usage, but weak ({outside_acc*100:.1f}%)")
        else:
            print(f"\n✗ NO STATE USAGE: Model relies only on SWA ({outside_acc*100:.1f}%)")
            if inside_acc > 0.5:
                print(f"    (Inside-window works at {inside_acc*100:.1f}% → SWA is doing the work)")
    
    results['inside_accuracy'] = inside_acc
    results['outside_accuracy'] = outside_acc
    results['uses_state'] = outside_acc > 0.5
    return results


def diagnose_state_mechanism(model, verbose=True):
    """
    Deep diagnostic of the GDN state mechanism.
    
    Tests:
    1. Does the state actually get written? (non-zero state)
    2. Does querying with the same key retrieve the value?
    3. Does the SWA retrieval path work with the state?
    
    This is a unit-test level diagnostic, not an end-to-end NIAH test.
    """
    model.eval()
    device = next(model.parameters()).device
    cfg = model.cfg
    H, K, V = cfg.n_heads, cfg.head_dim, cfg.value_dim
    
    results = {
        'state_written': False,
        'self_retrieval_works': False,
        'swa_retrieval_works': False,
        'issues': [],
    }
    
    with torch.no_grad():
        # === Test 1: Does state get written? ===
        # Create a simple sequence
        seq = torch.randint(100, 1000, (1, 64), device=device)
        _, _, diags, final_state = model(seq)
        
        state_norm = final_state.norm().item()
        state_max = final_state.abs().max().item()
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"STATE MECHANISM DIAGNOSTIC")
            print(f"{'='*60}")
            print(f"\n1. STATE WRITING")
            print(f"   State norm: {state_norm:.4f}")
            print(f"   State max:  {state_max:.4f}")
        
        if state_norm > 0.01:
            results['state_written'] = True
            if verbose:
                print(f"   ✓ State is being written")
        else:
            results['issues'].append("State norm near zero - nothing being stored")
            if verbose:
                print(f"   ✗ State is essentially zero!")
        
        # === Test 2: Self-retrieval (query state with same key that wrote it) ===
        # Find the first GDN layer
        gdn_layer = None
        for layer in model.layers:
            if hasattr(layer, 'k_proj') and hasattr(layer, 'v_proj'):
                gdn_layer = layer
                break
        
        if gdn_layer is not None:
            # Pick a specific token and compute its key/value
            test_token = torch.tensor([[500]], device=device)
            test_embed = model.embed(test_token)
            test_norm = gdn_layer.norm(test_embed)
            
            test_key = gdn_layer.k_proj(test_norm).view(1, 1, H, K)
            test_key = F.normalize(test_key.float(), p=2, dim=-1)
            test_val = gdn_layer.v_proj(test_norm).view(1, 1, H, V)
            
            # Create a sequence with TWO tokens to test shifted value mode
            # Token 0 = key token (500), Token 1 = value token (501)
            # In shifted mode: k from pos 0, v from pos 1
            # So querying with key(500) should retrieve value(501)
            seq_pair = torch.tensor([[500, 501]], device=device)
            
            # Manually run through just the first GDN layer
            x_pair = model.embed(seq_pair)
            out_pair, state_pair, _ = gdn_layer(x_pair, initial_state=None)
            
            # Check if using shifted value mode
            use_shifted = getattr(gdn_layer, 'use_shifted_value', False)
            
            if use_shifted:
                # In shifted mode: query with key(token 0) should retrieve value(token 1)
                key_token = torch.tensor([[500]], device=device)
                val_token = torch.tensor([[501]], device=device)
                
                key_emb = model.embed(key_token)
                val_emb = model.embed(val_token)
                
                key = gdn_layer.k_proj(gdn_layer.norm(key_emb)).view(H, K)
                key = F.normalize(key.float(), p=2, dim=-1)
                expected = gdn_layer.v_proj(gdn_layer.norm(val_emb)).view(H, V).float()
                
                test_name = "SHIFTED-VALUE RETRIEVAL (key=token0, expect=value_of_token1)"
            else:
                # Original mode: query with key should retrieve its own value
                key = test_key[0, 0].float()  # [H, K]
                expected = test_val[0, 0].float()  # [H, V]
                test_name = "SELF-RETRIEVAL (key=value from same token)"
            
            # Query state with key
            retrieved = torch.einsum('hkv,hk->hv', state_pair[0].float(), key)  # [H, V]
            
            # Compute cosine similarity per head
            sims = []
            for h in range(H):
                sim = F.cosine_similarity(retrieved[h].unsqueeze(0), expected[h].unsqueeze(0)).item()
                sims.append(sim)
            avg_sim = sum(sims) / len(sims)
            
            # Also check magnitude ratio
            ret_norms = [retrieved[h].norm().item() for h in range(H)]
            exp_norms = [expected[h].norm().item() for h in range(H)]
            ratios = [r/e if e > 1e-6 else 0 for r, e in zip(ret_norms, exp_norms)]
            
            if verbose:
                print(f"\n2. {test_name}")
                print(f"   Shifted value mode: {use_shifted}")
                print(f"   Avg cosine similarity: {avg_sim:.4f}")
                print(f"   Per-head sims: {[f'{s:.2f}' for s in sims]}")
                print(f"   Magnitude ratios (ret/exp): {[f'{r:.2f}' for r in ratios]}")
            
            if avg_sim > 0.9:  # Should be ~1.0 for proper retrieval
                results['self_retrieval_works'] = True
                if verbose:
                    print(f"   ✓ Retrieval works!")
            elif avg_sim > 0.5:
                results['issues'].append(f"Retrieval partial (sim={avg_sim:.4f})")
                if verbose:
                    print(f"   ⚠ Retrieval partial (expected ~1.0)")
            else:
                results['issues'].append(f"Retrieval fails (sim={avg_sim:.4f})")
                if verbose:
                    print(f"   ✗ Retrieval FAILS")
        
        # === Test 3: SWA retrieval path ===
        # Find the SWA layer
        swa_layer = None
        for layer in model.layers:
            if hasattr(layer, 'global_q_proj'):
                swa_layer = layer
                break
        
        if swa_layer is not None and results['state_written']:
            # Get a token embedding and project through global_q_proj
            test_embed = model.embed(torch.tensor([[500]], device=device))
            test_norm = swa_layer.norm(test_embed)
            
            q_g = swa_layer.global_q_proj(test_norm).view(1, 1, H, K)
            q_g = F.relu(q_g)  # As in the actual SWA forward
            
            # Check if query vector is non-trivial
            q_norm = q_g.norm().item()
            q_nnz = (q_g > 0).float().mean().item()  # Fraction of non-zero after ReLU
            
            if verbose:
                print(f"\n3. SWA RETRIEVAL PATH")
                print(f"   Query norm (after ReLU): {q_norm:.4f}")
                print(f"   Query sparsity: {(1-q_nnz)*100:.1f}% zeros")
            
            if q_norm > 0.1 and q_nnz > 0.1:
                # Try retrieval
                q_g_expanded = q_g.transpose(1, 2).float()  # [B, H, T, K]
                retrieved = torch.einsum('bhkv,bhtk->bhtv', final_state.float(), q_g_expanded)
                ret_norm = retrieved.norm().item()
                
                if verbose:
                    print(f"   Retrieved norm: {ret_norm:.4f}")
                
                if ret_norm > 0.01:
                    results['swa_retrieval_works'] = True
                    if verbose:
                        print(f"   ✓ SWA retrieval produces non-zero output")
                else:
                    results['issues'].append("SWA retrieval produces zero output")
                    if verbose:
                        print(f"   ✗ SWA retrieval is zero!")
            else:
                results['issues'].append(f"SWA query too sparse/weak (norm={q_norm:.4f})")
                if verbose:
                    print(f"   ✗ Query is too sparse/weak for retrieval")
        
        # === Test 4: Multi-token interference ===
        # Place multiple tokens and check if each can be retrieved
        if gdn_layer is not None:
            n_test_tokens = 10
            test_tokens = list(range(500, 500 + n_test_tokens))
            
            # Compute keys and values for each
            token_keys = []
            token_vals = []
            for tok in test_tokens:
                tok_t = torch.tensor([[tok]], device=device)
                tok_emb = model.embed(tok_t)
                tok_norm = gdn_layer.norm(tok_emb)
                
                k = gdn_layer.k_proj(tok_norm).view(H, K)
                k = F.normalize(k.float(), p=2, dim=-1)
                v = gdn_layer.v_proj(tok_norm).view(H, V).float()
                token_keys.append(k)
                token_vals.append(v)
            
            # Run sequence with all tokens through GDN layer
            seq_multi = torch.tensor([test_tokens], device=device)  # [1, 10]
            x_multi = model.embed(seq_multi)
            _, state_multi, _ = gdn_layer(x_multi, initial_state=None)
            
            # Query for each token
            multi_sims = []
            for i, (key, expected) in enumerate(zip(token_keys, token_vals)):
                retrieved = torch.einsum('hkv,hk->hv', state_multi[0].float(), key)
                
                # Per-head cosine sim, then average
                head_sims = []
                for h in range(H):
                    sim = F.cosine_similarity(retrieved[h].unsqueeze(0), expected[h].unsqueeze(0)).item()
                    head_sims.append(sim)
                multi_sims.append(sum(head_sims) / len(head_sims))
            
            avg_multi_sim = sum(multi_sims) / len(multi_sims)
            first_sim = multi_sims[0]  # First token (most overwritten)
            last_sim = multi_sims[-1]  # Last token (most recent)
            
            if verbose:
                print(f"\n4. MULTI-TOKEN INTERFERENCE ({n_test_tokens} tokens)")
                print(f"   Avg similarity across all tokens: {avg_multi_sim:.4f}")
                print(f"   First token (most overwritten): {first_sim:.4f}")
                print(f"   Last token (most recent): {last_sim:.4f}")
                print(f"   Per-token: {[f'{s:.2f}' for s in multi_sims]}")
            
            if avg_multi_sim > 0.7:
                results['multi_token_works'] = True
                if verbose:
                    print(f"   ✓ Multi-token retrieval works!")
            elif last_sim > 0.8 and first_sim < 0.3:
                results['issues'].append(f"Recency bias: last={last_sim:.2f}, first={first_sim:.2f}")
                if verbose:
                    print(f"   ⚠ State has RECENCY BIAS - old tokens overwritten")
            else:
                results['issues'].append(f"Multi-token fails (avg={avg_multi_sim:.4f})")
                if verbose:
                    print(f"   ✗ Multi-token retrieval fails")
        
        # === Summary ===
        if verbose:
            print(f"\n{'='*60}")
            print(f"SUMMARY")
            print(f"{'='*60}")
            all_good = all([results['state_written'], results['self_retrieval_works'], results['swa_retrieval_works']])
            if all_good:
                print(f"✓ All mechanisms working - architecture CAN retrieve from state")
            else:
                print(f"✗ Issues found:")
                for issue in results['issues']:
                    print(f"   - {issue}")
    
    model.train()
    return results


def test_state_density(model, seq_len=1024, max_needles=7, n_trials=30):
    """
    Find the saturation point of the state matrix.
    
    Incrementally adds needles until accuracy drops below threshold.
    Returns: maximum number of facts the state can reliably hold.
    """
    print(f"\n{'='*60}")
    print(f"STATE DENSITY TEST: Finding saturation point")
    print(f"{'='*60}")
    
    threshold = 0.7  # Require 70% accuracy to consider "reliable"
    
    results = []
    for n_needles in range(1, max_needles + 1):
        print(f"\nTesting {n_needles} needle(s)...")
        result = multi_needle_test(model, seq_len=seq_len, n_needles=n_needles, 
                                   n_trials=n_trials, verbose=False)
        results.append({
            'n_needles': n_needles,
            'accuracy': result['overall_accuracy'],
            'correct': result['total_correct'],
            'total': result['total_trials'],
        })
        print(f"  → {result['overall_accuracy']*100:.1f}% accuracy")
        
        if result['overall_accuracy'] < threshold:
            print(f"\n  Saturation detected at {n_needles} needles (below {threshold*100:.0f}% threshold)")
            break
    
    # Find max reliable capacity
    reliable = [r for r in results if r['accuracy'] >= threshold]
    max_capacity = reliable[-1]['n_needles'] if reliable else 0
    
    print(f"\n{'='*60}")
    print(f"RESULT: State can reliably hold {max_capacity} facts")
    print(f"{'='*60}")
    print(f"\nFull results:")
    print(f"  Needles  Accuracy")
    print(f"  -------  --------")
    for r in results:
        status = "✓" if r['accuracy'] >= threshold else "✗"
        print(f"  {r['n_needles']:7d}  {r['accuracy']*100:6.1f}%  {status}")
    
    return {
        'max_capacity': max_capacity,
        'results': results,
        'threshold': threshold,
    }


def run_full_diagnostic(model, seq_len=128, needle_pos=32):
    """Comprehensive diagnostic with state health check."""
    model.eval()
    device = next(model.parameters()).device
    cfg = model.cfg
    
    needle_id = cfg.vocab_size - 3
    seq = torch.randint(0, cfg.vocab_size - 100, (1, seq_len), device=device)
    seq[0, needle_pos] = cfg.marker_token
    seq[0, needle_pos + 1] = needle_id
    seq[0, -1] = cfg.cue_token
    
    with torch.no_grad():
        logits, _, diags, state = model(seq, return_diagnostics=True)
    
    print(f"\n{'='*60}")
    print("DIAGNOSTIC REPORT")
    print(f"{'='*60}")
    
    print(f"\nState Health:")
    print(f"  State norm: {state.norm().item():.2f}")
    print(f"  State max:  {state.abs().max().item():.2f}")
    
    if state.abs().max().item() < 10:
        print(f"  ✓ State bounded")
    elif state.abs().max().item() < 100:
        print(f"  ⚠ State moderately large")
    else:
        print(f"  ✗ State explosion!")
    
    print(f"\nLayer Diagnostics:")
    for i, d in enumerate(diags):
        if d['layer'] == 'G':
            print(f"  Layer {i} (GDN): β={d['beta_mean']:.3f}, g={d['g_mean']:.3f}, state={d['state_norm']:.2f}")
        else:
            print(f"  Layer {i} (SWA): gate={d['gate_mean']:.3f}, local={d['local_norm']:.2f}, retrieval={d['retrieval_norm']:.2f}")
    
    pred = logits[0, -1].argmax().item()
    print(f"\nPrediction: {pred} (target: {needle_id}) - {'✓' if pred == needle_id else '✗'}")
    
    return state, diags


# =============================================================================
# DELTA RULE VALIDATION
# =============================================================================

def validate_delta_rule(device='cuda'):
    """Validate TRUE Delta Rule properties."""
    print(f"\n{'='*60}")
    print("DELTA RULE VALIDATION")
    print(f"{'='*60}")
    
    results = {}
    
    # Test 1: Identical tokens
    print("\n1. Identical Tokens (Redundancy Suppression):")
    B, H, K, V = 1, 4, 32, 64
    state = torch.zeros(B, H, K, V, device=device)
    k = F.normalize(torch.randn(B, H, K, device=device), dim=-1)
    v = torch.randn(B, H, V, device=device)
    
    pred1 = torch.einsum('bhkv,bhk->bhv', state, k)
    error1 = v - pred1
    state = state + torch.einsum('bhv,bhk->bhkv', error1, k)
    norm1 = state.norm().item()
    
    pred2 = torch.einsum('bhkv,bhk->bhv', state, k)
    error2 = v - pred2
    state = state + torch.einsum('bhv,bhk->bhkv', error2, k)
    norm2 = state.norm().item()
    
    passed = error2.norm().item() < 0.001
    print(f"  Error2: {error2.norm().item():.6f} (should be ~0)")
    print(f"  Growth: {norm2/norm1:.4f}x")
    print(f"  → {'✓ PASS' if passed else '✗ FAIL'}")
    results['identical_tokens'] = passed
    
    # Test 2: Orthogonal keys
    print("\n2. Orthogonal Keys (Independent Storage):")
    state = torch.zeros(1, 1, 32, 64, device=device)
    k1 = torch.zeros(1, 1, 32, device=device); k1[0,0,0] = 1.0
    k2 = torch.zeros(1, 1, 32, device=device); k2[0,0,1] = 1.0
    v1 = torch.randn(1, 1, 64, device=device)
    v2 = torch.randn(1, 1, 64, device=device)
    
    state = state + torch.einsum('bhv,bhk->bhkv', v1, k1)
    state = state + torch.einsum('bhv,bhk->bhkv', v2, k2)
    
    ret1 = torch.einsum('bhkv,bhk->bhv', state, k1)
    ret2 = torch.einsum('bhkv,bhk->bhv', state, k2)
    err1 = (ret1 - v1).norm().item() / v1.norm().item()
    err2 = (ret2 - v2).norm().item() / v2.norm().item()
    
    passed = err1 < 0.001 and err2 < 0.001
    print(f"  v1 error: {err1:.6f}")
    print(f"  v2 error: {err2:.6f}")
    print(f"  → {'✓ PASS' if passed else '✗ FAIL'}")
    results['orthogonal_keys'] = passed
    
    # Test 3: Capacity
    print("\n3. Capacity (100 writes):")
    state = torch.zeros(1, 1, 32, 64, device=device)
    keys, values = [], []
    for i in range(100):
        k = F.normalize(torch.randn(1, 1, 32, device=device), dim=-1)
        v = torch.randn(1, 1, 64, device=device)
        keys.append(k); values.append(v)
        pred = torch.einsum('bhkv,bhk->bhv', state, k)
        error = v - pred
        state = state + torch.einsum('bhv,bhk->bhkv', error, k)
    
    ret_first = torch.einsum('bhkv,bhk->bhv', state, keys[0])
    ret_last = torch.einsum('bhkv,bhk->bhv', state, keys[-1])
    err_first = (ret_first - values[0]).norm().item() / values[0].norm().item()
    err_last = (ret_last - values[-1]).norm().item() / values[-1].norm().item()
    
    print(f"  State norm: {state.norm().item():.2f}")
    print(f"  First error: {err_first:.4f}")
    print(f"  Last error: {err_last:.4f}")
    print(f"  → First degrades (expected)")
    results['capacity'] = True
    
    print(f"\n{'='*60}")
    all_pass = all(v for v in results.values() if isinstance(v, bool))
    print(f"OVERALL: {'✓ ALL PASS' if all_pass else '✗ SOME FAILED'}")
    
    return results


# =============================================================================
# TRAINING
# =============================================================================

class TextDataset(Dataset):
    """Simple dataset for language modeling."""
    def __init__(self, tokens, seq_len=128):
        self.tokens = tokens
        self.seq_len = seq_len
        
    def __len__(self):
        return (len(self.tokens) - 1) // self.seq_len
    
    def __getitem__(self, idx):
        start = idx * self.seq_len
        return torch.tensor(self.tokens[start:start + self.seq_len + 1], dtype=torch.long)


def load_wikitext(n_tokens=500_000, seq_len=128, batch_size=16):
    """Load wikitext data for training."""
    from datasets import load_dataset
    from transformers import AutoTokenizer
    
    print(f"Loading {n_tokens:,} tokens from wikitext-103...")
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
    
    all_tokens = []
    for item in dataset:
        if item['text'].strip():
            all_tokens.extend(tokenizer.encode(item['text']))
            if len(all_tokens) >= n_tokens:
                break
    
    all_tokens = all_tokens[:n_tokens]
    print(f"Loaded {len(all_tokens):,} tokens")
    
    ds = TextDataset(all_tokens, seq_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


def compute_retrieval_loss(model, seq_len=256, batch_size=4):
    """
    Synthetic retrieval task for gradient signal.
    
    IMPORTANT: Uses VARIED needle IDs across FULL vocab range.
    This prevents the model from learning "predict high-vocab after CUE".
    """
    device = next(model.parameters()).device
    cfg = model.cfg
    
    tokens = torch.randint(0, cfg.vocab_size - 100, (batch_size, seq_len), device=device)
    targets = torch.full((batch_size, seq_len), -100, device=device)
    
    for i in range(batch_size):
        # CRITICAL: Use needle from WIDE range (100 to vocab_size-100)
        # This prevents shortcut learning of "predict high vocab tokens"
        needle_id = torch.randint(100, cfg.vocab_size - 100, (1,)).item()
        
        # Vary position more widely
        pos = torch.randint(10, seq_len // 2, (1,)).item()
        tokens[i, pos] = cfg.marker_token
        tokens[i, pos + 1] = needle_id
        tokens[i, -1] = cfg.cue_token
        
        targets[i, -1] = needle_id
    
    _, loss, _, _ = model(tokens, targets=targets)
    return loss


def compute_key_orthogonality_loss(model, sample_tokens, n_sample=64):
    """
    Regularization loss to encourage diverse keys.
    
    When keys for different tokens are orthogonal:
    - Writes don't interfere with each other
    - State can hold multiple independent facts
    
    Loss = mean(|cosine(k_i, k_j)|) for i != j
    Target: minimize to encourage orthogonality
    """
    device = next(model.parameters()).device
    cfg = model.cfg
    
    # Find GDN layer
    gdn_layer = None
    for layer in model.layers:
        if hasattr(layer, 'k_proj'):
            gdn_layer = layer
            break
    
    if gdn_layer is None:
        return torch.tensor(0.0, device=device)
    
    # Sample tokens from the batch
    B, T = sample_tokens.shape
    if T > n_sample:
        # Random sample of positions
        indices = torch.randperm(T, device=device)[:n_sample]
        tokens = sample_tokens[:1, indices]  # Use first batch item
    else:
        tokens = sample_tokens[:1]
    
    n = tokens.shape[1]
    
    # Compute keys
    emb = model.embed(tokens)
    x_norm = gdn_layer.norm(emb)
    keys = gdn_layer.k_proj(x_norm).view(1, n, cfg.n_heads, cfg.head_dim)
    keys = F.normalize(keys.float(), p=2, dim=-1)  # [1, n, H, K]
    
    # Compute average |cosine similarity| across all heads
    total_loss = 0.0
    for h in range(cfg.n_heads):
        k_h = keys[0, :, h, :]  # [n, K]
        sim_matrix = k_h @ k_h.T  # [n, n]
        
        # Mask out diagonal (self-similarity = 1)
        mask = ~torch.eye(n, dtype=torch.bool, device=device)
        off_diag = sim_matrix[mask]
        
        # Loss = mean of squared similarities (penalize high similarity)
        total_loss = total_loss + off_diag.pow(2).mean()
    
    return total_loss / cfg.n_heads


def compute_beta_sparsity_loss(model, sample_tokens):
    """
    Regularization loss to encourage sparse beta (selective writing).
    
    For good memory:
    - Most tokens should have LOW beta (don't store noise)
    - Only important tokens should have HIGH beta
    
    Loss = mean(beta) - encourages lower beta on average
    """
    device = next(model.parameters()).device
    cfg = model.cfg
    
    # Find GDN layer
    gdn_layer = None
    for layer in model.layers:
        if hasattr(layer, 'beta_proj'):
            gdn_layer = layer
            break
    
    if gdn_layer is None:
        return torch.tensor(0.0, device=device)
    
    # Compute beta for sample tokens
    emb = model.embed(sample_tokens)
    x_norm = gdn_layer.norm(emb)
    beta = torch.sigmoid(gdn_layer.beta_proj(x_norm))  # [B, T, H]
    
    # Loss = mean beta (lower is more selective)
    return beta.mean()


def train_with_regularization(
    model, 
    data_loader, 
    steps=1000, 
    warmup_steps=200,
    lr=3e-4, 
    retrieval_weight=2.0, 
    log_interval=100
):
    """
    Curriculum training: retrieval warmup → mixed LM/retrieval.
    
    Phase 1 (warmup): Pure retrieval loss
    Phase 2 (mixed): LM loss + weighted retrieval loss
    """
    device = next(model.parameters()).device
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    lm_iter = iter(data_loader)
    history = {'step': [], 'lm': [], 'ret': [], 'phase': []}
    
    print(f"Training {steps} steps ({warmup_steps} warmup)")
    print(f"  LR: {lr}, Retrieval weight: {retrieval_weight}")
    print("="*60)
    
    model.train()
    start_time = time.time()
    
    for step in range(steps):
        optimizer.zero_grad()
        
        if step < warmup_steps:
            ret_loss = compute_retrieval_loss(model)
            ret_loss.backward()
            history['ret'].append(ret_loss.item())
            history['lm'].append(0)
            history['phase'].append('warmup')
        else:
            try:
                batch = next(lm_iter)
            except StopIteration:
                lm_iter = iter(data_loader)
                batch = next(lm_iter)
            
            input_ids = batch[:, :-1].to(device)
            targets = batch[:, 1:].to(device)
            _, lm_loss, _, _ = model(input_ids, targets)
            
            ret_loss = compute_retrieval_loss(model)
            
            total = lm_loss + retrieval_weight * ret_loss
            total.backward()
            
            history['lm'].append(lm_loss.item())
            history['ret'].append(ret_loss.item())
            history['phase'].append('mixed')
        
        history['step'].append(step)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if step % log_interval == 0:
            phase = "WARMUP" if step < warmup_steps else "MIXED"
            lm = history['lm'][-1]
            ret = history['ret'][-1]
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed
            print(f"[{phase:6s}] Step {step:5d}: LM={lm:6.3f} RET={ret:6.3f} ({steps_per_sec:.1f} steps/s)")
    
    total_time = time.time() - start_time
    print("="*60)
    print(f"Training complete: {steps} steps in {total_time:.1f}s")
    
    return history


def train_with_key_reg(
    model, 
    data_loader, 
    steps=1000,
    lr=3e-4, 
    key_orth_weight=0.1,
    beta_sparsity_weight=0.1,
    retrieval_weight=1.0,
    log_interval=50
):
    """
    Training with regularization to preserve associative memory capability.
    
    Losses:
    1. LM loss: Language modeling (next-token prediction)
    2. Retrieval loss: NIAH synthetic task
    3. Key orthogonality: Encourage diverse keys (prevent collapse)
    4. Beta sparsity: Encourage selective writing (prevent overwriting)
    
    The key insight: Standard LM training collapses keys (all similar),
    which destroys the state's ability to hold multiple facts.
    """
    device = next(model.parameters()).device
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    lm_iter = iter(data_loader)
    history = {'step': [], 'lm': [], 'ret': [], 'key_orth': [], 'beta_sparse': [], 'total': []}
    
    print(f"Training with regularization ({steps} steps)")
    print(f"  LR: {lr}")
    print(f"  Weights: retrieval={retrieval_weight}, key_orth={key_orth_weight}, beta_sparse={beta_sparsity_weight}")
    print("="*60)
    
    model.train()
    start_time = time.time()
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # Get batch
        try:
            batch = next(lm_iter)
        except StopIteration:
            lm_iter = iter(data_loader)
            batch = next(lm_iter)
        
        input_ids = batch[:, :-1].to(device)
        targets = batch[:, 1:].to(device)
        
        # Forward pass for LM loss
        _, lm_loss, _, _ = model(input_ids, targets)
        
        # Regularization losses
        key_orth_loss = compute_key_orthogonality_loss(model, input_ids)
        beta_sparse_loss = compute_beta_sparsity_loss(model, input_ids)
        
        # Retrieval loss (every N steps to save compute)
        if step % 5 == 0:
            ret_loss = compute_retrieval_loss(model)
        else:
            ret_loss = torch.tensor(0.0, device=device)
        
        # Total loss
        total_loss = (
            lm_loss 
            + retrieval_weight * ret_loss
            + key_orth_weight * key_orth_loss
            + beta_sparsity_weight * beta_sparse_loss
        )
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Log
        history['step'].append(step)
        history['lm'].append(lm_loss.item())
        history['ret'].append(ret_loss.item() if ret_loss.item() > 0 else history['ret'][-1] if history['ret'] else 0)
        history['key_orth'].append(key_orth_loss.item())
        history['beta_sparse'].append(beta_sparse_loss.item())
        history['total'].append(total_loss.item())
        
        if step % log_interval == 0:
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed
            print(f"Step {step:4d}: LM={lm_loss.item():.3f}, KO={key_orth_loss.item():.3f}, "
                  f"BS={beta_sparse_loss.item():.3f}, RET={history['ret'][-1]:.3f} ({steps_per_sec:.1f} s/s)")
    
    total_time = time.time() - start_time
    print("="*60)
    print(f"Training complete: {steps} steps in {total_time:.1f}s")
    
    # Final stats
    print(f"\nFinal metrics:")
    print(f"  Key orthogonality: {history['key_orth'][-1]:.4f} (lower = more orthogonal)")
    print(f"  Beta sparsity: {history['beta_sparse'][-1]:.4f} (lower = more selective)")
    
    return history


def analyze_gradients(model, seq_len=64, verbose=True):
    """Analyze gradient flow through the model."""
    device = next(model.parameters()).device
    model.train()
    
    x = torch.randint(0, model.cfg.vocab_size - 100, (2, seq_len), device=device)
    marker_pos = seq_len // 4
    cue_pos = seq_len - 1
    x[:, marker_pos] = model.cfg.marker_token
    x[:, marker_pos + 1] = model.cfg.vocab_size - 3
    x[:, cue_pos] = model.cfg.cue_token
    
    targets = torch.full((2, seq_len), -100, device=device)
    targets[:, -1] = model.cfg.vocab_size - 3
    
    model.zero_grad()
    _, loss, _, _ = model(x, targets)
    loss.backward()
    
    grad_info = {}
    has_nan = False
    has_inf = False
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            grad_info[name] = grad_norm
            if torch.isnan(param.grad).any():
                has_nan = True
            if torch.isinf(param.grad).any():
                has_inf = True
    
    if verbose:
        print(f"\n{'='*60}")
        print("GRADIENT ANALYSIS")
        print(f"{'='*60}")
        print(f"Loss: {loss.item():.4f}")
        print(f"NaN: {'✗ YES' if has_nan else '✓ NO'}")
        print(f"Inf: {'✗ YES' if has_inf else '✓ NO'}")
    
    return {'grad_norms': grad_info, 'has_nan': has_nan, 'has_inf': has_inf, 'loss': loss.item()}


# =============================================================================
# MEMORY GYM: Synthetic Associative Recall (SAR)
# =============================================================================

class MemoryGymDataset(Dataset):
    """
    Synthetic Associative Recall dataset for training selective memory.
    
    Each sequence contains N key-value pairs scattered in noise, followed by
    a query for ONE specific key. The model must learn to:
    1. Store each (marker_type, value) pair distinctly
    2. Ignore the "haystack" noise between pairs
    3. Retrieve the specific value when queried
    
    This teaches the GDN to:
    - Keep β ≈ 0 for noise (don't write)
    - Spike β ≈ 1 for needles (write important)
    - Use distinct key representations for each marker type
    """
    
    def __init__(
        self, 
        n_samples: int = 10000,
        seq_len: int = 1024,
        n_needles: int = 3,
        vocab_size: int = 32000,
        marker_base: int = None,  # Will use vocab_size - 50 if None
        value_base: int = None,   # Will use vocab_size - 100 if None
        noise_range: int = None,  # Will use vocab_size - 200 if None
    ):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.n_needles = n_needles
        self.vocab_size = vocab_size
        
        # Token ranges (non-overlapping)
        self.marker_base = marker_base or (vocab_size - 50)
        self.value_base = value_base or (vocab_size - 100)
        self.noise_range = noise_range or (vocab_size - 200)
        self.cue_token = vocab_size - 1
        
        # Pre-generate all samples for speed
        self.samples = []
        self.labels = []
        self._generate_samples()
    
    def _generate_samples(self):
        """Generate all training samples."""
        for _ in range(self.n_samples):
            seq, label = self._make_one_sample()
            self.samples.append(seq)
            self.labels.append(label)
    
    def _make_one_sample(self):
        """
        Create one SAR sequence:
        [noise] [MARKER_0, VALUE_0] [noise] [MARKER_1, VALUE_1] ... [CUE, QUERY_MARKER]
        
        Label: The VALUE corresponding to the queried MARKER
        """
        seq = torch.randint(0, self.noise_range, (self.seq_len,))
        
        # Distribute needles evenly across sequence (leaving room at end for query)
        positions = []
        usable_len = self.seq_len - 4  # Leave space for [CUE, QUERY_MARKER, ?, ?]
        for i in range(self.n_needles):
            pos = int((i + 1) * usable_len / (self.n_needles + 1))
            positions.append(pos)
        
        # Place needles
        needle_values = []
        for i, pos in enumerate(positions):
            marker = self.marker_base + i  # Unique marker per needle type
            value = self.value_base + torch.randint(0, 30, (1,)).item()  # Random value
            needle_values.append(value)
            
            seq[pos] = marker
            seq[pos + 1] = value
        
        # Choose which needle to query (random)
        query_idx = torch.randint(0, self.n_needles, (1,)).item()
        query_marker = self.marker_base + query_idx
        expected_value = needle_values[query_idx]
        
        # Place query at end: [CUE, QUERY_MARKER]
        seq[-2] = self.cue_token
        seq[-1] = query_marker
        
        return seq, expected_value
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]


def create_memory_gym_loader(
    n_samples: int = 10000,
    seq_len: int = 1024,
    n_needles: int = 3,
    batch_size: int = 4,
    vocab_size: int = 32000,
):
    """
    Create a DataLoader for Memory Gym training.
    
    Args:
        n_samples: Number of training samples
        seq_len: Sequence length
        n_needles: Number of facts per sequence
        batch_size: Batch size
        vocab_size: Model vocabulary size
    
    Returns:
        DataLoader with (input_seq, target_value) pairs
    """
    dataset = MemoryGymDataset(
        n_samples=n_samples,
        seq_len=seq_len,
        n_needles=n_needles,
        vocab_size=vocab_size,
    )
    
    def collate_fn(batch):
        seqs = torch.stack([b[0] for b in batch])
        labels = torch.tensor([b[1] for b in batch])
        return seqs, labels
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    print(f"Memory Gym: {n_samples} samples, T={seq_len}, {n_needles} needles, batch={batch_size}")
    return loader


def train_memory_gym(
    model,
    n_samples: int = 10000,
    seq_len: int = 1024,
    n_needles: int = 3,
    batch_size: int = 4,
    steps: int = 500,
    lr: float = 3e-4,
    log_interval: int = 50,
):
    """
    Train the model on Memory Gym (Synthetic Associative Recall).
    
    This curriculum teaches the model to:
    1. Distinguish between multiple marker types
    2. Store values selectively (high β for needles, low β for noise)
    3. Retrieve the correct value based on the query marker
    
    Returns:
        Training history with loss curves and accuracy
    """
    device = next(model.parameters()).device
    model.train()
    
    # Create loader
    loader = create_memory_gym_loader(
        n_samples=n_samples,
        seq_len=seq_len,
        n_needles=n_needles,
        batch_size=batch_size,
        vocab_size=model.cfg.vocab_size,
    )
    
    optimizer = AdamW(model.parameters(), lr=lr)
    
    history = {
        'step': [],
        'loss': [],
        'accuracy': [],
    }
    
    data_iter = iter(loader)
    start_time = time.time()
    
    print("="*60)
    print(f"MEMORY GYM TRAINING: {n_needles} needles, T={seq_len}")
    print("="*60)
    
    for step in range(steps):
        try:
            seqs, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            seqs, labels = next(data_iter)
        
        seqs = seqs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        logits, _, _, _ = model(seqs)
        
        # Loss: predict the correct value at the last position
        # logits[:, -1] are the predictions after seeing [CUE, QUERY_MARKER]
        last_logits = logits[:, -1]  # [B, vocab_size]
        loss = F.cross_entropy(last_logits, labels)
        
        # Accuracy
        preds = last_logits.argmax(dim=-1)
        accuracy = (preds == labels).float().mean().item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        history['step'].append(step)
        history['loss'].append(loss.item())
        history['accuracy'].append(accuracy)
        
        if step % log_interval == 0:
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed
            avg_acc = sum(history['accuracy'][-log_interval:]) / min(log_interval, len(history['accuracy']))
            print(f"Step {step:4d}: Loss={loss.item():.4f}, Acc={avg_acc*100:.1f}%, ({steps_per_sec:.1f} steps/s)")
    
    total_time = time.time() - start_time
    final_acc = sum(history['accuracy'][-100:]) / min(100, len(history['accuracy']))
    
    print("="*60)
    print(f"Training complete: {steps} steps in {total_time:.1f}s")
    print(f"Final accuracy (last 100): {final_acc*100:.1f}%")
    print("="*60)
    
    return history


def train_mixed_curriculum(
    model,
    wikitext_loader,
    gym_seq_len: int = 1024,
    n_needles: int = 3,
    steps: int = 1000,
    gym_ratio: float = 0.3,
    lr: float = 3e-4,
    log_interval: int = 100,
):
    """
    Three-phase curriculum mimicking neural development:
    
    Phase 1 (10%): Retrieval warmup - basic storage/recall
    Phase 2 (30%): Pure Gym - selective multi-needle storage  
    Phase 3 (60%): Mixed LM + Gym - language while maintaining selectivity
    
    Progression: Simple → Specialized → Complex
    """
    device = next(model.parameters()).device
    model.train()
    
    # Auto-calculate phase boundaries
    retrieval_steps = int(steps * 0.10)  # 10% retrieval warmup
    gym_steps = int(steps * 0.30)        # 30% pure gym
    mixed_steps = steps - retrieval_steps - gym_steps  # 60% mixed
    
    # Create Memory Gym loader
    gym_loader = create_memory_gym_loader(
        n_samples=5000,
        seq_len=gym_seq_len,
        n_needles=n_needles,
        batch_size=4,
        vocab_size=model.cfg.vocab_size,
    )
    
    optimizer = AdamW(model.parameters(), lr=lr)
    
    history = {
        'step': [],
        'lm_loss': [],
        'gym_loss': [],
        'gym_accuracy': [],
        'phase': [],
    }
    
    wiki_iter = iter(wikitext_loader)
    gym_iter = iter(gym_loader)
    start_time = time.time()
    
    print("="*60)
    print(f"THREE-PHASE CURRICULUM ({n_needles} needles)")
    print(f"  Phase 1: {retrieval_steps} steps - Retrieval warmup")
    print(f"  Phase 2: {gym_steps} steps - Pure Gym (selectivity)")
    print(f"  Phase 3: {mixed_steps} steps - Mixed ({(1-gym_ratio)*100:.0f}% LM, {gym_ratio*100:.0f}% Gym)")
    print("="*60)
    
    for step in range(steps):
        optimizer.zero_grad()
        
        # Determine phase
        if step < retrieval_steps:
            phase = 'retrieval'
        elif step < retrieval_steps + gym_steps:
            phase = 'gym'
        else:
            phase = 'mixed'
        
        # Execute based on phase
        if phase == 'retrieval':
            # Pure retrieval (single-needle) - basic storage
            ret_loss = compute_retrieval_loss(model)
            ret_loss.backward()
            history['lm_loss'].append(None)
            history['gym_loss'].append(ret_loss.item())
            history['gym_accuracy'].append(None)
            history['phase'].append('retrieval')
            
        elif phase == 'gym':
            # Pure Gym - multi-needle selectivity
            try:
                seqs, labels = next(gym_iter)
            except StopIteration:
                gym_iter = iter(gym_loader)
                seqs, labels = next(gym_iter)
            
            seqs = seqs.to(device)
            labels = labels.to(device)
            
            logits, _, _, _ = model(seqs)
            last_logits = logits[:, -1]
            loss = F.cross_entropy(last_logits, labels)
            
            preds = last_logits.argmax(dim=-1)
            accuracy = (preds == labels).float().mean().item()
            
            loss.backward()
            history['lm_loss'].append(None)
            history['gym_loss'].append(loss.item())
            history['gym_accuracy'].append(accuracy)
            history['phase'].append('gym')
            
        else:  # mixed phase
            do_gym = torch.rand(1).item() < gym_ratio
            
            if do_gym:
                try:
                    seqs, labels = next(gym_iter)
                except StopIteration:
                    gym_iter = iter(gym_loader)
                    seqs, labels = next(gym_iter)
                
                seqs = seqs.to(device)
                labels = labels.to(device)
                
                logits, _, _, _ = model(seqs)
                last_logits = logits[:, -1]
                loss = F.cross_entropy(last_logits, labels)
                
                preds = last_logits.argmax(dim=-1)
                accuracy = (preds == labels).float().mean().item()
                
                loss.backward()
                history['gym_loss'].append(loss.item())
                history['gym_accuracy'].append(accuracy)
                history['lm_loss'].append(None)
                history['phase'].append('mixed-gym')
            else:
                # Language modeling step
                try:
                    batch = next(wiki_iter)
                except StopIteration:
                    wiki_iter = iter(wikitext_loader)
                    batch = next(wiki_iter)
                
                x = batch.to(device)
                _, loss, _, _ = model(x, x)
                
                loss.backward()
                history['lm_loss'].append(loss.item())
                history['gym_loss'].append(None)
                history['gym_accuracy'].append(None)
                history['phase'].append('mixed-lm')
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        history['step'].append(step)
        
        if step % log_interval == 0:
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed
            
            # Average recent metrics
            recent_lm = [x for x in history['lm_loss'][-log_interval:] if x is not None]
            recent_gym = [x for x in history['gym_loss'][-log_interval:] if x is not None]
            recent_acc = [x for x in history['gym_accuracy'][-log_interval:] if x is not None]
            
            lm_avg = sum(recent_lm) / len(recent_lm) if recent_lm else 0
            gym_avg = sum(recent_gym) / len(recent_gym) if recent_gym else 0
            acc_avg = sum(recent_acc) / len(recent_acc) if recent_acc else 0
            
            phase_label = phase.upper()
            print(f"[{phase_label:8s}] Step {step:4d}: LM={lm_avg:.3f}, Gym={gym_avg:.3f}, GymAcc={acc_avg*100:.1f}% ({steps_per_sec:.1f} steps/s)")

    total_time = time.time() - start_time
    final_gym_acc = [x for x in history['gym_accuracy'][-100:] if x is not None]
    final_acc = sum(final_gym_acc) / len(final_gym_acc) if final_gym_acc else 0
    
    print("="*60)
    print(f"Training complete: {steps} steps in {total_time:.1f}s")
    print(f"Final Gym accuracy: {final_acc*100:.1f}%")
    print("="*60)
    
    return history


# =============================================================================
# PROGRESSIVE CURRICULUM: Retrieval → Gym → LM (accuracy-gated)
# =============================================================================

def train_progressive_curriculum(
    model,
    wikitext_loader=None,
    max_steps: int = 3000,
    lr: float = 3e-4,
    log_interval: int = 50,
    # Phase thresholds (transition when accuracy exceeds)
    retrieval_threshold: float = 0.95,
    gym_threshold: float = 0.70,
    # Gym progression
    gym_start_needles: int = 2,
    gym_max_needles: int = 5,
    gym_start_seq: int = 256,
    gym_max_seq: int = 2048,
    # Safety limits
    max_retrieval_steps: int = 200,
    max_gym_steps: int = 2000,
):
    """
    Progressive curriculum with accuracy-gated phase transitions.
    
    Order: Retrieval → Gym → LM (NO LM until gym succeeds)
    
    Phase 1 (Retrieval): Single-needle until accuracy > retrieval_threshold
    Phase 2 (Gym): Multi-needle with progressive difficulty:
        - Start: 2 needles, short seq
        - Increase difficulty as accuracy improves
        - Continue until multi-needle accuracy > gym_threshold
    Phase 3 (LM): Only after gym proves selective gating works
    
    This teaches β gating BEFORE introducing language noise.
    """
    device = next(model.parameters()).device
    model.train()
    
    optimizer = AdamW(model.parameters(), lr=lr)
    
    history = {
        'step': [],
        'phase': [],
        'loss': [],
        'accuracy': [],
        'n_needles': [],
        'seq_len': [],
        'beta_mean': [],
    }
    
    # State tracking
    phase = 'retrieval'
    current_needles = gym_start_needles
    current_seq_len = gym_start_seq
    phase_start_step = 0
    recent_accuracies = []
    
    # Create initial gym loader
    gym_loader = None
    gym_iter = None
    wiki_iter = iter(wikitext_loader) if wikitext_loader else None
    
    start_time = time.time()
    
    print("="*60)
    print("PROGRESSIVE CURRICULUM")
    print("="*60)
    print(f"Phase 1: Retrieval until {retrieval_threshold*100:.0f}% (max {max_retrieval_steps} steps)")
    print(f"Phase 2: Gym {gym_start_needles}→{gym_max_needles} needles until {gym_threshold*100:.0f}%")
    print(f"Phase 3: LM (only after gym succeeds)")
    print("="*60)
    
    def get_beta_stats(model):
        """Get mean β value from GDN layers."""
        betas = []
        for layer in model.layers:
            if hasattr(layer, 'beta_proj'):
                # Sample a forward pass to get β
                with torch.no_grad():
                    x = torch.randint(0, 1000, (1, 64), device=device)
                    emb = model.embed(x)
                    # Get β from projection (bias is already in beta_proj)
                    beta = torch.sigmoid(layer.beta_proj(layer.norm(emb)))
                    betas.append(beta.mean().item())
        return sum(betas) / len(betas) if betas else 0.0
    
    def check_needle_survival(model, n_needles, seq_len=256):
        """
        Beacon tracker: Check if needles survive in state after full sequence.
        
        Properly queries the state using the GDN layer's key projection,
        then measures cosine similarity between retrieved and expected values.
        
        Returns: (mean_survival, per_needle_survivals)
        """
        model.eval()
        with torch.no_grad():
            # Create a test sequence with known needles
            vocab_size = model.cfg.vocab_size
            markers = list(range(1000, 1000 + n_needles))
            values = list(range(2000, 2000 + n_needles))
            
            # Build sequence: noise, needle1, noise, needle2, ...
            seq = torch.randint(100, 900, (1, seq_len), device=device)
            positions = []
            for i, (m, v) in enumerate(zip(markers, values)):
                pos = (i + 1) * seq_len // (n_needles + 2)
                seq[0, pos] = m
                seq[0, pos + 1] = v
                positions.append(pos)
            
            # Run sequence through model, get final state
            _, _, _, final_state = model(seq)  # state: [B, H, K, V]
            
            # Find the first GDN layer to use its projections
            gdn_layer = None
            for layer in model.layers:
                if hasattr(layer, 'k_proj') and hasattr(layer, 'v_proj'):
                    gdn_layer = layer
                    break
            
            if gdn_layer is None:
                model.train()
                return 0.0, [0.0] * n_needles
            
            # For each marker, check if its value is retrievable from state
            survivals = []
            H, K, V = model.cfg.n_heads, model.cfg.head_dim, model.cfg.value_dim
            
            for i, (m_tok, v_tok) in enumerate(zip(markers, values)):
                pos = positions[i]
                
                # Get the marker's embedding at its position (from actual sequence)
                marker_embed = model.embed(torch.tensor([[m_tok]], device=device))  # [1, 1, D]
                marker_norm = gdn_layer.norm(marker_embed)
                
                # Project to key space and L2 normalize (matching GDN forward)
                marker_key = gdn_layer.k_proj(marker_norm).view(1, 1, H, K)  # [1, 1, H, K]
                marker_key = F.normalize(marker_key.float(), p=2, dim=-1)  # [1, 1, H, K]
                
                # Query state: retrieved[h,v] = sum_k(state[h,k,v] * key[h,k])
                # state: [B, H, K, V] -> [1, H, K, V]
                # key: [1, 1, H, K] -> [H, K]
                key = marker_key[0, 0]  # [H, K]
                retrieved = torch.einsum('hkv,hk->hv', final_state[0].float(), key)  # [H, V]
                
                # Get expected value: what SHOULD be stored for this marker
                # At marker position, the value stored is v_proj(embed(marker))
                value_embed = model.embed(torch.tensor([[m_tok]], device=device))  # marker's own embedding
                value_norm = gdn_layer.norm(value_embed)
                expected_value = gdn_layer.v_proj(value_norm).view(1, 1, H, V)  # [1, 1, H, V]
                expected = expected_value[0, 0]  # [H, V]
                
                # Compare retrieved vs expected (per-head, then average)
                head_sims = []
                for h in range(H):
                    cos = F.cosine_similarity(
                        retrieved[h].unsqueeze(0), 
                        expected[h].unsqueeze(0)
                    ).item()
                    head_sims.append(cos)
                avg_sim = sum(head_sims) / len(head_sims)
                survivals.append(avg_sim)
            
            model.train()
            return sum(survivals) / len(survivals), survivals
    
    def create_gym_loader_for_difficulty(n_needles, seq_len):
        return create_memory_gym_loader(
            n_samples=2000,
            seq_len=seq_len,
            n_needles=n_needles,
            batch_size=4,
            vocab_size=model.cfg.vocab_size,
        )
    
    for step in range(max_steps):
        optimizer.zero_grad()
        
        # =====================================================================
        # PHASE 1: RETRIEVAL (single-needle)
        # =====================================================================
        if phase == 'retrieval':
            ret_loss = compute_retrieval_loss(model)
            ret_loss.backward()
            
            # Estimate accuracy from loss (retrieval is CE loss)
            accuracy = max(0, 1 - ret_loss.item() / 10)  # Rough estimate
            
            history['loss'].append(ret_loss.item())
            history['accuracy'].append(accuracy)
            history['n_needles'].append(1)
            history['seq_len'].append(64)
            
            recent_accuracies.append(accuracy)
            if len(recent_accuracies) > 20:
                recent_accuracies.pop(0)
            
            # Check for phase transition
            steps_in_phase = step - phase_start_step
            avg_acc = sum(recent_accuracies) / len(recent_accuracies)
            
            if avg_acc > retrieval_threshold or steps_in_phase >= max_retrieval_steps:
                print(f"\n[TRANSITION] Retrieval → Gym at step {step} (acc={avg_acc*100:.1f}%)")
                phase = 'gym'
                phase_start_step = step
                recent_accuracies = []
                gym_loader = create_gym_loader_for_difficulty(current_needles, current_seq_len)
                gym_iter = iter(gym_loader)
        
        # =====================================================================
        # PHASE 2: GYM (multi-needle, progressive difficulty)
        # =====================================================================
        elif phase == 'gym':
            try:
                seqs, labels = next(gym_iter)
            except StopIteration:
                gym_iter = iter(gym_loader)
                seqs, labels = next(gym_iter)
            
            seqs = seqs.to(device)
            labels = labels.to(device)
            
            logits, _, _, _ = model(seqs)
            last_logits = logits[:, -1]
            loss = F.cross_entropy(last_logits, labels)
            
            preds = last_logits.argmax(dim=-1)
            accuracy = (preds == labels).float().mean().item()
            
            loss.backward()
            
            history['loss'].append(loss.item())
            history['accuracy'].append(accuracy)
            history['n_needles'].append(current_needles)
            history['seq_len'].append(current_seq_len)
            
            recent_accuracies.append(accuracy)
            if len(recent_accuracies) > 50:
                recent_accuracies.pop(0)
            
            # Check for difficulty increase
            avg_acc = sum(recent_accuracies) / len(recent_accuracies) if recent_accuracies else 0
            
            if len(recent_accuracies) >= 50 and avg_acc > 0.80:
                # Increase difficulty
                if current_needles < gym_max_needles:
                    current_needles += 1
                    recent_accuracies = []
                    gym_loader = create_gym_loader_for_difficulty(current_needles, current_seq_len)
                    gym_iter = iter(gym_loader)
                    print(f"\n[DIFFICULTY] Increased to {current_needles} needles at step {step}")
                elif current_seq_len < gym_max_seq:
                    current_seq_len = min(current_seq_len * 2, gym_max_seq)
                    recent_accuracies = []
                    gym_loader = create_gym_loader_for_difficulty(current_needles, current_seq_len)
                    gym_iter = iter(gym_loader)
                    print(f"\n[DIFFICULTY] Increased to seq_len={current_seq_len} at step {step}")
            
            # Check for phase transition to LM
            steps_in_phase = step - phase_start_step
            if (current_needles >= gym_max_needles and 
                current_seq_len >= gym_max_seq and 
                avg_acc > gym_threshold):
                print(f"\n[TRANSITION] Gym → LM at step {step} (acc={avg_acc*100:.1f}%)")
                phase = 'lm'
                phase_start_step = step
                recent_accuracies = []
            elif steps_in_phase >= max_gym_steps:
                print(f"\n[TRANSITION] Gym → LM at step {step} (max steps, acc={avg_acc*100:.1f}%)")
                phase = 'lm'
                phase_start_step = step
                recent_accuracies = []
        
        # =====================================================================
        # PHASE 3: LM (language modeling, only after gym succeeds)
        # =====================================================================
        elif phase == 'lm':
            if wiki_iter is None:
                print("\n[WARNING] No WikiText loader provided, ending training.")
                break
            
            try:
                batch = next(wiki_iter)
            except StopIteration:
                wiki_iter = iter(wikitext_loader)
                batch = next(wiki_iter)
            
            x = batch.to(device)
            _, loss, _, _ = model(x, x)
            loss.backward()
            
            history['loss'].append(loss.item())
            history['accuracy'].append(None)
            history['n_needles'].append(None)
            history['seq_len'].append(x.shape[1])
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        history['step'].append(step)
        history['phase'].append(phase)
        history['beta_mean'].append(get_beta_stats(model))
        
        # Logging
        if step % log_interval == 0:
            elapsed = time.time() - start_time
            steps_per_sec = (step + 1) / elapsed
            
            recent_loss = history['loss'][-log_interval:]
            avg_loss = sum(recent_loss) / len(recent_loss) if recent_loss else 0
            
            recent_acc = [x for x in history['accuracy'][-log_interval:] if x is not None]
            avg_acc = sum(recent_acc) / len(recent_acc) if recent_acc else 0
            
            beta_mean = history['beta_mean'][-1]
            
            if phase == 'gym':
                # Beacon check: are needles surviving in state?
                survival, per_needle = check_needle_survival(model, current_needles, current_seq_len)
                surv_str = ','.join([f'{s:.2f}' for s in per_needle])
                print(f"[{phase.upper():4s}] Step {step:4d}: loss={avg_loss:.3f}, acc={avg_acc*100:.1f}%, "
                      f"needles={current_needles}, seq={current_seq_len}, β={beta_mean:.3f}, surv=[{surv_str}] ({steps_per_sec:.1f} s/s)")
            elif phase == 'retrieval':
                print(f"[{phase.upper():4s}] Step {step:4d}: loss={avg_loss:.3f}, β={beta_mean:.3f} ({steps_per_sec:.1f} s/s)")
            else:
                print(f"[{phase.upper():4s}] Step {step:4d}: loss={avg_loss:.3f}, β={beta_mean:.3f} ({steps_per_sec:.1f} s/s)")
    
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print(f"Training complete: {step+1} steps in {total_time:.1f}s")
    print(f"Final phase: {phase}")
    print(f"Final β mean: {history['beta_mean'][-1]:.3f}")
    print("="*60)
    
    return history
