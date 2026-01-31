#!/usr/bin/env python3
"""
Step 2: Validate Triton Backward Kernel
Place this in deepseek-bones/ next to gt-v6-gdn-bwd-kernel.py
Run: python deepseek-bones/step2_test_backward.py
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
import sys
import os

print("=" * 70)
print("STEP 2: TRITON BACKWARD KERNEL VALIDATION")
print("=" * 70)
print(f"PyTorch: {torch.__version__}")
print(f"Triton:  {triton.__version__}")
print(f"GPU:     {torch.cuda.get_device_name(0)}")
print("=" * 70)

# =============================================================================
# REFERENCE IMPLEMENTATION
# =============================================================================

def reference_delta_rule(k, v, beta, g, initial_state=None):
    B, T, H, K_dim = k.shape
    V_dim = v.shape[-1]
    
    if initial_state is None:
        state = torch.zeros(B, H, K_dim, V_dim, device=k.device, dtype=k.dtype)
    else:
        state = initial_state.clone()
    
    outputs = []
    for t in range(T):
        k_t = k[:, t]
        v_t = v[:, t]
        beta_t = beta[:, t]
        g_t = g[:, t]
        
        pred = torch.einsum('bhkv,bhk->bhv', state, k_t)
        error = v_t - pred
        update = torch.einsum('bhv,bhk->bhkv', error, k_t)
        state = g_t[..., None, None] * state + beta_t[..., None, None] * update
        outputs.append(torch.einsum('bhkv,bhk->bhv', state, k_t))
    
    return torch.stack(outputs, dim=1), state

# =============================================================================
# FORWARD KERNEL (needed by backward module)
# =============================================================================

@triton.jit
def delta_rule_fwd_kernel(
    K_ptr, V_ptr, Beta_ptr, G_ptr, State_ptr, Out_ptr,
    stride_k_b, stride_k_t, stride_k_h, stride_k_d,
    stride_v_b, stride_v_t, stride_v_h, stride_v_d,
    stride_bg_b, stride_bg_t, stride_bg_h,
    stride_s_b, stride_s_h, stride_s_k, stride_s_v,
    stride_o_b, stride_o_t, stride_o_h, stride_o_v,
    T,
    K_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    k_offs = tl.arange(0, K_DIM)
    v_offs = tl.arange(0, V_DIM)
    
    state_base = State_ptr + pid_b * stride_s_b + pid_h * stride_s_h
    state_ptrs = state_base + k_offs[:, None] * stride_s_k + v_offs[None, :] * stride_s_v
    state = tl.load(state_ptrs).to(tl.float32)
    
    for t in range(T):
        k_base = K_ptr + pid_b * stride_k_b + t * stride_k_t + pid_h * stride_k_h
        k_t = tl.load(k_base + k_offs * stride_k_d).to(tl.float32)
        
        v_base = V_ptr + pid_b * stride_v_b + t * stride_v_t + pid_h * stride_v_h
        v_t = tl.load(v_base + v_offs * stride_v_d).to(tl.float32)
        
        bg_offset = pid_b * stride_bg_b + t * stride_bg_t + pid_h * stride_bg_h
        beta_t = tl.load(Beta_ptr + bg_offset).to(tl.float32)
        g_t = tl.load(G_ptr + bg_offset).to(tl.float32)
        
        pred = tl.sum(state * k_t[:, None], axis=0)
        error = v_t - pred
        outer = k_t[:, None] * error[None, :]
        state = g_t * state + beta_t * outer
        
        out_t = tl.sum(state * k_t[:, None], axis=0)
        out_base = Out_ptr + pid_b * stride_o_b + t * stride_o_t + pid_h * stride_o_h
        tl.store(out_base + v_offs * stride_o_v, out_t)
    
    tl.store(state_ptrs, state)


def triton_delta_rule(k, v, beta, g, initial_state=None):
    B, T, H, K_DIM = k.shape
    V_DIM = v.shape[-1]
    device = k.device
    orig_dtype = k.dtype
    
    k = k.contiguous().float()
    v = v.contiguous().float()
    beta = beta.contiguous().float()
    g = g.contiguous().float()
    
    if initial_state is None:
        state = torch.zeros(B, H, K_DIM, V_DIM, device=device, dtype=torch.float32)
    else:
        state = initial_state.contiguous().float().clone()
    
    out = torch.empty(B, T, H, V_DIM, device=device, dtype=torch.float32)
    
    delta_rule_fwd_kernel[(B, H)](
        k, v, beta, g, state, out,
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        beta.stride(0), beta.stride(1), beta.stride(2),
        state.stride(0), state.stride(1), state.stride(2), state.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        T, K_DIM, V_DIM,
    )
    
    return out.to(orig_dtype), state.to(orig_dtype)

# =============================================================================
# FIND BACKWARD KERNEL
# =============================================================================

print("\n[1] Locating backward kernel...")

script_dir = os.path.dirname(os.path.abspath(__file__))

search_paths = [
    os.path.join(script_dir, 'gt-v6-gdn-bwd-kernel.py'),
    'gt-v6-gdn-bwd-kernel.py',
    'deepseek-bones/gt-v6-gdn-bwd-kernel.py',
]

bwd_path = None
for p in search_paths:
    if os.path.exists(p):
        bwd_path = os.path.abspath(p)
        break

if not bwd_path:
    print("ERROR: Can't find gt-v6-gdn-bwd-kernel.py")
    print("Searched:", search_paths)
    sys.exit(1)

print(f"Found: {bwd_path}")

# =============================================================================
# LOAD MODULE
# =============================================================================

print("\n[2] Loading module...")

module_dir = os.path.dirname(bwd_path)
if module_dir not in sys.path:
    sys.path.insert(0, module_dir)

module_ns = {
    '__name__': '__not_main__', 
    '__file__': bwd_path,
    'triton_delta_rule': triton_delta_rule,  # Inject forward kernel
    'torch': torch,
    'triton': triton,
    'tl': tl,
    'F': F,
}
with open(bwd_path, 'r') as f:
    code = f.read()

exec(compile(code, bwd_path, 'exec'), module_ns)
print("Module loaded")

# =============================================================================
# TEST
# =============================================================================

print("\n[3] Testing backward kernel...")

B, T, H, K, V = 2, 32, 4, 16, 32
torch.manual_seed(42)

if 'delta_rule_autograd' in module_ns:
    delta_rule_autograd = module_ns['delta_rule_autograd']
elif 'DeltaRuleFunction' in module_ns:
    DeltaRuleFunction = module_ns['DeltaRuleFunction']
    def delta_rule_autograd(k, v, beta, g, state):
        return DeltaRuleFunction.apply(k, v, beta, g, state)
else:
    print("ERROR: No delta_rule_autograd or DeltaRuleFunction found")
    print("Found:", [k for k in module_ns.keys() if not k.startswith('_')])
    sys.exit(1)

# Create tensors
k = F.normalize(torch.randn(B, T, H, K, device='cuda'), dim=-1).requires_grad_(True)
v = torch.randn(B, T, H, V, device='cuda').requires_grad_(True)
beta = torch.sigmoid(torch.randn(B, T, H, device='cuda') - 2).requires_grad_(True)
g = torch.sigmoid(torch.randn(B, T, H, device='cuda') + 2).requires_grad_(True)
state_init = torch.randn(B, H, K, V, device='cuda').requires_grad_(True)

# Reference
out_ref, state_ref = reference_delta_rule(k, v, beta, g, state_init)
(out_ref.sum() + state_ref.sum()).backward()

ref_grads = {
    'k': k.grad.clone(),
    'v': v.grad.clone(),
    'beta': beta.grad.clone(),
    'g': g.grad.clone(),
    'state_init': state_init.grad.clone(),
}

# Zero
k.grad = v.grad = beta.grad = g.grad = state_init.grad = None

# Triton
out_tri, state_tri = delta_rule_autograd(k, v, beta, g, state_init)
(out_tri.sum() + state_tri.sum()).backward()

# Compare
print(f"\n{'Param':<12} {'Ref':<12} {'Triton':<12} {'MaxDiff':<12} {'RelErr':<12}")
print("-" * 65)

tensors = {'k': k, 'v': v, 'beta': beta, 'g': g, 'state_init': state_init}
all_ok = True

for name, tensor in tensors.items():
    ref = ref_grads[name]
    tri = tensor.grad
    
    if tri is None:
        print(f"{name:<12} MISSING GRADIENT")
        all_ok = False
        continue
    
    r_norm = ref.norm().item()
    t_norm = tri.norm().item()
    diff = (tri - ref).abs().max().item()
    rel = diff / (r_norm + 1e-8)
    
    ok = rel < 1e-3
    print(f"{name:<12} {r_norm:<12.4f} {t_norm:<12.4f} {diff:<12.6f} {rel:<12.6f} {'✓' if ok else '✗'}")
    if not ok:
        all_ok = False

print(f"\n→ {'✓ BACKWARD KERNEL VALIDATED' if all_ok else '✗ NEEDS DEBUG'}")

if all_ok:
    print("\nYour Triton backward kernel is correct. Ready for Step 3.")
