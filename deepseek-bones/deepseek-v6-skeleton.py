# =============================================================================
# DELTA RULE INTEGRATION TEST SUITE & ARCHITECTURE
# =============================================================================
# 
# This module provides:
# 1. Stable Triton Delta Rule kernel (already validated)
# 2. Comprehensive test suite for integration
# 3. Modular architecture compatible with existing builds
# 4. Performance analysis tools
#
# Key Principles:
# - Maintain backward compatibility with existing modules
# - No simplification to sequential loops
# - Preserve all original design patterns
# - Full validation at every integration point
# =============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
import time
import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import warnings

# =============================================================================
# 1. STABLE TRITON KERNEL (UNCHANGED)
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
    """
    Stable Delta Rule kernel - NO changes to avoid breaking existing builds.
    Each program handles one (batch, head) pair.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    k_offs = tl.arange(0, K_DIM)
    v_offs = tl.arange(0, V_DIM)
    
    state_base = State_ptr + pid_b * stride_s_b + pid_h * stride_s_h
    state_ptrs = state_base + k_offs[:, None] * stride_s_k + v_offs[None, :] * stride_s_v
    
    state = tl.load(state_ptrs).to(tl.float32)
    
    for t in range(T):
        # Load k[t]
        k_base = K_ptr + pid_b * stride_k_b + t * stride_k_t + pid_h * stride_k_h
        k_t = tl.load(k_base + k_offs * stride_k_d).to(tl.float32)
        
        # Load v[t]
        v_base = V_ptr + pid_b * stride_v_b + t * stride_v_t + pid_h * stride_v_h
        v_t = tl.load(v_base + v_offs * stride_v_d).to(tl.float32)
        
        # Load scalars
        bg_offset = pid_b * stride_bg_b + t * stride_bg_t + pid_h * stride_bg_h
        beta_t = tl.load(Beta_ptr + bg_offset).to(tl.float32)
        g_t = tl.load(G_ptr + bg_offset).to(tl.float32)
        
        # Prediction
        pred = tl.sum(state * k_t[:, None], axis=0)
        error = v_t - pred
        
        # Outer product and update
        outer = k_t[:, None] * error[None, :]
        state = g_t * state + beta_t * outer
        
        # Output
        out_t = tl.sum(state * k_t[:, None], axis=0)
        
        # Store output
        out_base = Out_ptr + pid_b * stride_o_b + t * stride_o_t + pid_h * stride_o_h
        tl.store(out_base + v_offs * stride_o_v, out_t)
    
    tl.store(state_ptrs, state)


def triton_delta_rule(k: torch.Tensor, 
                     v: torch.Tensor, 
                     beta: torch.Tensor, 
                     g: torch.Tensor, 
                     initial_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper for Triton Delta Rule - maintains exact same API as before.
    
    Args:
        k: [B, T, H, K_DIM] - normalized keys
        v: [B, T, H, V_DIM] - values
        beta: [B, T, H] - write gate
        g: [B, T, H] - forget gate
        initial_state: [B, H, K_DIM, V_DIM] or None
        
    Returns:
        output: [B, T, H, V_DIM]
        final_state: [B, H, K_DIM, V_DIM]
    """
    B, T, H, K_DIM = k.shape
    V_DIM = v.shape[-1]
    device = k.device
    orig_dtype = k.dtype
    
    # Ensure contiguous
    k = k.contiguous().float()
    v = v.contiguous().float()
    beta = beta.contiguous().float()
    g = g.contiguous().float()
    
    if initial_state is None:
        state = torch.zeros(B, H, K_DIM, V_DIM, device=device, dtype=torch.float32)
    else:
        state = initial_state.contiguous().float().clone()
    
    out = torch.empty(B, T, H, V_DIM, device=device, dtype=torch.float32)
    
    grid = (B, H)
    
    delta_rule_fwd_kernel[grid](
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
# 2. COMPREHENSIVE TEST SUITE
# =============================================================================

class DeltaRuleTestSuite:
    """Complete test suite for Delta Rule integration"""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        self.results = {}
        
    def test_kernel_correctness(self, 
                               B: int = 2, 
                               T: int = 64, 
                               H: int = 4, 
                               K: int = 32, 
                               V: int = 64) -> Dict[str, Any]:
        """
        Test Triton kernel against PyTorch reference implementation.
        Returns detailed error metrics.
        """
        print(f"\n{'='*60}")
        print(f"KERNEL CORRECTNESS TEST: B={B}, T={T}, H={H}, K={K}, V={V}")
        print(f"{'='*60}")
        
        # Generate test data
        torch.manual_seed(42)
        k = F.normalize(torch.randn(B, T, H, K, device=self.device), dim=-1)
        v = torch.randn(B, T, H, V, device=self.device)
        beta = torch.sigmoid(torch.randn(B, T, H, device=self.device) - 2)
        g = torch.sigmoid(torch.randn(B, T, H, device=self.device) + 2)
        
        # Reference implementation (preserves original logic)
        def reference_delta_rule(k, v, beta, g, initial_state=None):
            B, T, H, K_dim = k.shape
            V_dim = v.shape[-1]
            state = torch.zeros(B, H, K_dim, V_dim, device=k.device) if initial_state is None else initial_state.clone()
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
        
        # Run both implementations
        with torch.no_grad():
            out_ref, state_ref = reference_delta_rule(k, v, beta, g)
            out_tri, state_tri = triton_delta_rule(k, v, beta, g)
        
        # Calculate errors
        out_abs_err = (out_tri - out_ref).abs().max().item()
        out_rel_err = ((out_tri - out_ref).norm() / out_ref.norm()).item()
        
        state_abs_err = (state_tri - state_ref).abs().max().item()
        state_rel_err = ((state_tri - state_ref).norm() / state_ref.norm()).item()
        
        # Store results
        test_results = {
            'output_abs_error': out_abs_err,
            'output_rel_error': out_rel_err,
            'state_abs_error': state_abs_err,
            'state_rel_error': state_rel_err,
            'passed': out_abs_err < 1e-5 and state_abs_err < 1e-5,
            'config': {'B': B, 'T': T, 'H': H, 'K': K, 'V': V}
        }
        
        # Print results
        print(f"Output absolute error: {out_abs_err:.2e}")
        print(f"Output relative error: {out_rel_err:.2e}")
        print(f"State absolute error:  {state_abs_err:.2e}")
        print(f"State relative error:  {state_rel_err:.2e}")
        print(f"✓ PASS" if test_results['passed'] else "✗ FAIL")
        
        self.results['kernel_correctness'] = test_results
        return test_results
    
    def test_performance(self, 
                        configs: Optional[list] = None,
                        n_warmup: int = 5,
                        n_runs: int = 20) -> Dict[str, Any]:
        """
        Performance benchmarking across multiple configurations.
        """
        if configs is None:
            configs = [
                (4, 64, 8, 32, 64),
                (8, 128, 8, 32, 64),
                (8, 256, 8, 32, 64),
                (16, 128, 8, 32, 64),
            ]
        
        print(f"\n{'='*60}")
        print("PERFORMANCE BENCHMARK")
        print(f"{'='*60}")
        print(f"{'Configuration':>25} | {'PyTorch (ms)':>12} | {'Triton (ms)':>11} | {'Speedup':>8}")
        print(f"{'-'*70}")
        
        performance_results = {}
        
        for B, T, H, K, V in configs:
            # Generate data
            k = F.normalize(torch.randn(B, T, H, K, device=self.device), dim=-1)
            v = torch.randn(B, T, H, V, device=self.device)
            beta = torch.sigmoid(torch.randn(B, T, H, device=self.device) - 2)
            g = torch.sigmoid(torch.randn(B, T, H, device=self.device) + 2)
            
            # Reference timing
            def ref_func():
                state = torch.zeros(B, H, K, V, device=self.device)
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
                torch.cuda.synchronize()
            
            # Warmup and time reference
            for _ in range(n_warmup):
                ref_func()
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(n_runs):
                ref_func()
            torch.cuda.synchronize()
            ref_time = (time.perf_counter() - start) / n_runs * 1000
            
            # Warmup and time Triton
            for _ in range(n_warmup):
                triton_delta_rule(k, v, beta, g)
                torch.cuda.synchronize()
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(n_runs):
                triton_delta_rule(k, v, beta, g)
            torch.cuda.synchronize()
            tri_time = (time.perf_counter() - start) / n_runs * 1000
            
            speedup = ref_time / tri_time
            config_str = f"B={B},T={T},H={H},K={K},V={V}"
            
            print(f"{config_str:>25} | {ref_time:>11.2f}ms | {tri_time:>10.2f}ms | {speedup:>7.2f}x")
            
            performance_results[config_str] = {
                'pytorch_ms': ref_time,
                'triton_ms': tri_time,
                'speedup': speedup,
                'config': {'B': B, 'T': T, 'H': H, 'K': K, 'V': V}
            }
        
        self.results['performance'] = performance_results
        return performance_results
    
    def test_gradient_flow(self, 
                          B: int = 2, 
                          T: int = 32, 
                          H: int = 4, 
                          K: int = 16, 
                          V: int = 32) -> Dict[str, Any]:
        """
        Test that gradients flow correctly through the Delta Rule.
        """
        print(f"\n{'='*60}")
        print("GRADIENT FLOW TEST")
        print(f"{'='*60}")
        
        # Create learnable parameters
        k_proj = nn.Linear(64, H * K).to(self.device)
        v_proj = nn.Linear(64, H * V).to(self.device)
        beta_proj = nn.Linear(64, H).to(self.device)
        g_proj = nn.Linear(64, H).to(self.device)
        
        # Input
        x = torch.randn(B, T, 64, device=self.device, requires_grad=True)
        
        # Forward pass
        k = k_proj(x).view(B, T, H, K)
        v = v_proj(x).view(B, T, H, V)
        beta = torch.sigmoid(beta_proj(x))
        g = torch.sigmoid(g_proj(x))
        
        k = F.normalize(k, dim=-1)
        
        # Apply Delta Rule
        output, state = triton_delta_rule(k, v, beta, g)
        
        # Loss and backward
        loss = output.sum()
        loss.backward()
        
        # Check gradients
        grad_checks = {
            'x_has_grad': x.grad is not None,
            'x_grad_norm': x.grad.norm().item() if x.grad is not None else 0,
            'k_proj_has_grad': k_proj.weight.grad is not None,
            'v_proj_has_grad': v_proj.weight.grad is not None,
            'no_nan_in_grads': not torch.isnan(x.grad).any().item() if x.grad is not None else True,
            'no_inf_in_grads': not torch.isinf(x.grad).any().item() if x.grad is not None else True,
        }
        
        grad_results = {
            **grad_checks,
            'passed': all(grad_checks.values()),
        }
        
        # Print results
        for key, value in grad_checks.items():
            status = "✓" if value else "✗"
            print(f"{status} {key}: {value}")
        
        print(f"→ {'✓ GRADIENTS FLOW CORRECTLY' if grad_results['passed'] else '✗ GRADIENT ISSUE'}")
        
        self.results['gradient_flow'] = grad_results
        return grad_results
    
    def test_memory_behavior(self, 
                           B: int = 4, 
                           T: int = 1024, 
                           H: int = 8, 
                           K: int = 32, 
                           V: int = 64) -> Dict[str, Any]:
        """
        Test memory usage and behavior with large sequences.
        """
        print(f"\n{'='*60}")
        print("MEMORY BEHAVIOR TEST (Large Sequence)")
        print(f"{'='*60}")
        
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1024**2
        
        # Generate large sequence
        k = F.normalize(torch.randn(B, T, H, K, device=self.device), dim=-1)
        v = torch.randn(B, T, H, V, device=self.device)
        beta = torch.sigmoid(torch.randn(B, T, H, device=self.device) - 2)
        g = torch.sigmoid(torch.randn(B, T, H, device=self.device) + 2)
        
        # Warmup
        for _ in range(3):
            triton_delta_rule(k, v, beta, g)
        
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        final_memory = torch.cuda.memory_allocated() / 1024**2
        
        memory_results = {
            'initial_memory_mb': initial_memory,
            'peak_memory_mb': peak_memory,
            'final_memory_mb': final_memory,
            'memory_increase_mb': peak_memory - initial_memory,
            'sequence_length': T,
            'total_elements': B * T * H * (K + V + 2),
        }
        
        print(f"Initial memory: {initial_memory:.1f} MB")
        print(f"Peak memory:    {peak_memory:.1f} MB")
        print(f"Memory increase: {peak_memory - initial_memory:.1f} MB")
        print(f"Sequence length: {T}")
        print(f"Total elements: {memory_results['total_elements']:,}")
        
        self.results['memory_behavior'] = memory_results
        return memory_results
    
    def run_full_suite(self) -> Dict[str, Any]:
        """
        Run all tests and return comprehensive results.
        """
        print(f"\n{'='*60}")
        print("DELTA RULE COMPREHENSIVE TEST SUITE")
        print(f"{'='*60}")
        
        # Run all tests
        kernel_test = self.test_kernel_correctness()
        performance_test = self.test_performance()
        gradient_test = self.test_gradient_flow()
        memory_test = self.test_memory_behavior()
        
        # Summary
        print(f"\n{'='*60}")
        print("TEST SUITE SUMMARY")
        print(f"{'='*60}")
        
        all_passed = (
            kernel_test['passed'] and
            gradient_test['passed']
        )
        
        summary = {
            'all_tests_passed': all_passed,
            'kernel_correctness': kernel_test['passed'],
            'gradient_flow': gradient_test['passed'],
            'performance_tested': True,
            'memory_tested': True,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': self.device,
            'triton_version': triton.__version__,
            'torch_version': torch.__version__,
            'cuda_version': torch.version.cuda,
        }
        
        for key, value in summary.items():
            status = "✓" if value else "✗"
            print(f"{status} {key}: {value}")
        
        # Save results
        self.save_results()
        
        return summary
    
    def save_results(self, path: str = "delta_rule_test_results.json"):
        """Save test results to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nResults saved to: {path}")


# =============================================================================
# 3. MODULAR ARCHITECTURE COMPONENTS
# =============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Normalization - compatible with existing builds."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / norm * self.weight


class GatedDeltaNetLayer(nn.Module):
    """
    Complete Delta Rule layer with proper integration points.
    Maintains backward compatibility with existing architectures.
    """
    def __init__(self, 
                 d_model: int, 
                 n_heads: int, 
                 head_dim: int, 
                 value_dim: int,
                 use_triton: bool = True):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.value_dim = value_dim
        self.use_triton = use_triton
        
        # Projections (maintain original naming)
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * value_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * value_dim, d_model, bias=False)
        
        # Gate projections
        self.beta_proj = nn.Linear(d_model, n_heads, bias=True)
        nn.init.constant_(self.beta_proj.bias, -2.0)
        
        self.g_proj = nn.Linear(d_model, n_heads, bias=True)
        nn.init.constant_(self.g_proj.bias, 2.0)
        
        # Normalization
        self.norm = RMSNorm(d_model)
        
        # State initialization
        self.state = None
        
    def init_state(self, batch_size: int, device: torch.device):
        """Initialize or reset state."""
        self.state = torch.zeros(
            batch_size, self.n_heads, self.head_dim, self.value_dim,
            device=device
        )
    
    def forward(self, 
                x: torch.Tensor, 
                initial_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional initial state.
        Compatible with existing call signatures.
        """
        B, T, D = x.shape
        H, K, V = self.n_heads, self.head_dim, self.value_dim
        
        # Normalization
        x_norm = self.norm(x)
        
        # Projections
        k = self.k_proj(x_norm).view(B, T, H, K)
        v = self.v_proj(x_norm).view(B, T, H, V)
        
        # Normalize keys
        k = F.normalize(k.float(), p=2, dim=-1).to(x.dtype)
        
        # Gates
        beta = torch.sigmoid(self.beta_proj(x_norm))
        g = torch.sigmoid(self.g_proj(x_norm))
        
        # Use initial state if provided, otherwise use stored state
        current_state = initial_state if initial_state is not None else self.state
        
        # Apply Delta Rule
        if self.use_triton:
            out, new_state = triton_delta_rule(k, v, beta, g, current_state)
        else:
            # Fallback to reference implementation (for debugging)
            warnings.warn("Using reference Delta Rule implementation (slow!)")
            out, new_state = self._reference_delta_rule(k, v, beta, g, current_state)
        
        # Update stored state
        self.state = new_state.detach() if self.training else new_state
        
        # Output projection with residual
        out = out.to(x.dtype).reshape(B, T, H * V)
        return x + self.o_proj(out), new_state
    
    def _reference_delta_rule(self, k, v, beta, g, initial_state=None):
        """Reference implementation for debugging."""
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
# 4. INTEGRATION VALIDATION
# =============================================================================

class DeltaRuleIntegrationValidator:
    """Validates Delta Rule integration with existing architectures."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
    def validate_with_existing_model(self, 
                                   model: nn.Module, 
                                   input_shape: Tuple[int, int, int],
                                   n_steps: int = 10) -> Dict[str, Any]:
        """
        Validate Delta Rule integration with an existing model.
        
        Args:
            model: Existing model that should use Delta Rule
            input_shape: (batch, seq_len, features)
            n_steps: Number of forward steps to test
        
        Returns:
            Validation results
        """
        print(f"\n{'='*60}")
        print("EXISTING MODEL INTEGRATION VALIDATION")
        print(f"{'='*60}")
        
        model = model.to(self.device)
        model.eval()
        
        results = {
            'forward_passes': [],
            'memory_usage': [],
            'state_consistency': True,
        }
        
        # Test sequential forward passes with state retention
        state = None
        for step in range(n_steps):
            x = torch.randn(*input_shape, device=self.device)
            
            # Forward pass
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            with torch.no_grad():
                output, new_state = model(x, initial_state=state)
            
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - start) * 1000
            
            # Record results
            step_result = {
                'step': step,
                'time_ms': elapsed,
                'output_shape': tuple(output.shape),
                'state_shape': tuple(new_state.shape) if new_state is not None else None,
                'output_mean': output.mean().item(),
                'output_std': output.std().item(),
            }
            
            results['forward_passes'].append(step_result)
            
            # Check state consistency
            if state is not None:
                state_changed = not torch.allclose(state, new_state, rtol=1e-5)
                if not state_changed:
                    results['state_consistency'] = False
                    print(f"⚠ Warning: State unchanged at step {step}")
            
            state = new_state
            
            print(f"Step {step}: {elapsed:.2f}ms, Output: {output.shape}, "
                  f"State: {new_state.shape if new_state is not None else 'None'}")
        
        # Check memory behavior
        torch.cuda.empty_cache()
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        results['peak_memory_mb'] = peak_memory
        
        print(f"\nPeak memory usage: {peak_memory:.1f} MB")
        print(f"State consistency: {'✓' if results['state_consistency'] else '✗'}")
        
        return results
    
    def compare_implementations(self, 
                              d_model: int = 256,
                              n_heads: int = 8,
                              head_dim: int = 32,
                              value_dim: int = 64,
                              seq_len: int = 128,
                              batch_size: int = 4) -> Dict[str, Any]:
        """
        Compare Triton vs reference implementations.
        """
        print(f"\n{'='*60}")
        print("IMPLEMENTATION COMPARISON")
        print(f"{'='*60}")
        
        # Create both implementations
        triton_layer = GatedDeltaNetLayer(
            d_model, n_heads, head_dim, value_dim, use_triton=True
        ).to(self.device)
        
        ref_layer = GatedDeltaNetLayer(
            d_model, n_heads, head_dim, value_dim, use_triton=False
        ).to(self.device)
        
        # Copy weights to ensure fair comparison
        ref_layer.load_state_dict(triton_layer.state_dict())
        
        # Test input
        x = torch.randn(batch_size, seq_len, d_model, device=self.device)
        
        # Warmup
        for _ in range(3):
            triton_layer(x)
            ref_layer(x)
        
        # Time Triton implementation
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(20):
            out_tri, state_tri = triton_layer(x)
        torch.cuda.synchronize()
        tri_time = (time.perf_counter() - start) / 20 * 1000
        
        # Time reference implementation
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(5):  # Fewer runs because it's slower
            out_ref, state_ref = ref_layer(x)
        torch.cuda.synchronize()
        ref_time = (time.perf_counter() - start) / 5 * 1000
        
        # Compare outputs
        output_diff = (out_tri - out_ref).abs().max().item()
        state_diff = (state_tri - state_ref).abs().max().item() if state_tri is not None and state_ref is not None else 0
        
        comparison = {
            'triton_time_ms': tri_time,
            'reference_time_ms': ref_time,
            'speedup': ref_time / tri_time,
            'output_max_diff': output_diff,
            'state_max_diff': state_diff,
            'outputs_match': output_diff < 1e-5,
            'states_match': state_diff < 1e-5,
        }
        
        print(f"Triton time:  {tri_time:.2f}ms")
        print(f"Reference time: {ref_time:.2f}ms")
        print(f"Speedup: {comparison['speedup']:.2f}x")
        print(f"Output max difference: {output_diff:.2e}")
        print(f"State max difference: {state_diff:.2e}")
        print(f"Implementations match: {'✓' if comparison['outputs_match'] and comparison['states_match'] else '✗'}")
        
        return comparison


# =============================================================================
# 5. MAIN EXECUTION & ANALYSIS
# =============================================================================

def main():
    """Main execution: run comprehensive tests and analysis."""
    print(f"\n{'='*80}")
    print("DELTA RULE INTEGRATION FRAMEWORK")
    print(f"{'='*80}")
    
    # Environment info
    print(f"\nEnvironment Information:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.version.cuda}")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"  Triton: {triton.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("⚠ Warning: CUDA not available. Some tests may fail.")
        return
    
    # Create test suite
    test_suite = DeltaRuleTestSuite(device='cuda')
    
    # Run comprehensive tests
    print(f"\n{'='*80}")
    print("RUNNING COMPREHENSIVE TEST SUITE")
    print(f"{'='*80}")
    
    summary = test_suite.run_full_suite()
    
    # Integration validation
    print(f"\n{'='*80}")
    print("RUNNING INTEGRATION VALIDATION")
    print(f"{'='*80}")
    
    validator = DeltaRuleIntegrationValidator(device='cuda')
    
    # Test with a sample model
    sample_model = GatedDeltaNetLayer(
        d_model=256,
        n_heads=8,
        head_dim=32,
        value_dim=64,
        use_triton=True
    )
    
    integration_results = validator.validate_with_existing_model(
        sample_model,
        input_shape=(4, 128, 256),
        n_steps=5
    )
    
    # Compare implementations
    comparison = validator.compare_implementations()
    
    # Final summary
    print(f"\n{'='*80}")
    print("FINAL INTEGRATION STATUS")
    print(f"{'='*80}")
    
    all_checks_passed = (
        summary['all_tests_passed'] and
        integration_results['state_consistency'] and
        comparison['outputs_match'] and
        comparison['states_match']
    )
    
    print(f"\nIntegration Status: {'✓ READY FOR PRODUCTION' if all_checks_passed else '✗ NEEDS FURTHER TESTING'}")
    print(f"\nNext Steps:")
    print(f"  1. Import the stable Triton kernel into your existing build")
    print(f"  2. Replace sequential Delta Rule implementations with triton_delta_rule()")
    print(f"  3. Ensure input shapes match: [B, T, H, K], [B, T, H, V], [B, T, H], [B, T, H]")
    print(f"  4. Test with your specific model configurations")
    print(f"  5. Monitor performance with the provided test suite")
    
    # Save all results
    final_results = {
        'test_suite': test_suite.results,
        'integration': integration_results,
        'comparison': comparison,
        'summary': summary,
        'environment': {
            'pytorch': torch.__version__,
            'cuda': torch.version.cuda,
            'triton': triton.__version__,
            'gpu': torch.cuda.get_device_name(0),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
    }
    
    with open("delta_rule_integration_report.json", 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"\nFull report saved to: delta_rule_integration_report.json")
    
    return all_checks_passed


if __name__ == "__main__":
    # Run the complete integration framework
    success = main()
    
    if success:
        print(f"\n{'='*80}")
        print("INTEGRATION SUCCESSFUL - You can now safely integrate into your build!")
        print(f"{'='*80}")
    else:
        print(f"\n{'='*80}")
        print("INTEGRATION ISSUES DETECTED - Review test results above")
        print(f"{'='*80}")