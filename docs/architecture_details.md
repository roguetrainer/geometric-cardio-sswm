# Architecture Details

## Encoder Architecture

```
Input: (B, N, 5) point cloud
  ↓
Point-wise MLP: (5 → 64 → 64 → 128)
  ↓
Max Pooling: (B, N, 128) → (B, 128)
  ↓
Global MLP: (128 → 64 → 32)
  ↓
Output: (B, 32) latent vector
```

## Dynamics Model

```
Input: z_t (B, 32), A_t (B, 3, 3)
  ↓
Flatten A_t: (B, 9)
  ↓
Concat: [z_t, A_t] → (B, 41)
  ↓
MLP: (41 → 64 → 64)
  ↓
GRU: (64) with hidden state
  ↓
Output MLP: (64 → 64 → 32)
  ↓
Residual: z_t+1 = z_t + α·Δz
```

## Decoder Architecture

```
Input: z_t+1 (B, 32)
  ↓
MLP: (32 → 128 → 256 → 512 → N×6)
  ↓
Reshape: (B, N×6) → (B, N, 6)
  ↓
Output: (B, N, 6) point cloud
```

## Loss Function

L_total = L_recon + β·L_KL + α·L_geom + γ·L_constraint

where:
- L_recon = MSE(Ô_t+1, O_t+1)
- L_KL = KL(q(z|x) || p(z))
- L_geom = geodesic regularization
- L_constraint = physics constraints
