from fla.ops.gated_delta_rule import chunk_gated_delta_rule
import torch

B, T, H, K, V = 2, 128, 8, 32, 64
q = torch.randn(B, H, T, K, device='cuda')
k = torch.randn(B, H, T, K, device='cuda')  
v = torch.randn(B, H, T, V, device='cuda')
beta = torch.sigmoid(torch.randn(B, H, T, device='cuda'))
g = torch.sigmoid(torch.randn(B, H, T, device='cuda'))

# Check signature
import inspect
print(inspect.signature(chunk_gated_delta_rule))

# Try forward
out, state = chunk_gated_delta_rule(q, k, v, beta, g, output_final_state=True)
print(f"Output: {out.shape}, State: {state.shape}")