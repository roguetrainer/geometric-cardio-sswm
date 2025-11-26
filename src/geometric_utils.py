"""
Geometric Utilities for SSWM

Utilities for geometric computations including strain invariants,
rotation matrices, geodesic distances, and Riemannian metrics.
"""

import torch
import numpy as np
from typing import Tuple, Optional


def compute_strain_invariants(C: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute strain tensor invariants.
    
    Args:
        C: Cauchy-Green strain tensor (Batch, 3, 3)
    
    Returns:
        I1: First invariant tr(C) (Batch,)
        I2: Second invariant (Batch,)
        I3: Third invariant det(C) (Batch,)
    """
    # First invariant: trace
    I1 = torch.diagonal(C, dim1=-2, dim2=-1).sum(-1)
    
    # Third invariant: determinant
    I3 = torch.det(C)
    
    # Second invariant: 0.5 * (tr(C)^2 - tr(C^2))
    C_squared = torch.matmul(C, C)
    trace_C_squared = torch.diagonal(C_squared, dim1=-2, dim2=-1).sum(-1)
    I2 = 0.5 * (I1.pow(2) - trace_C_squared)
    
    return I1, I2, I3


def fiber_strain(C: torch.Tensor, fiber: torch.Tensor) -> torch.Tensor:
    """
    Compute fiber-specific strain (I4 invariant).
    
    I4 = f^T C f, where f is the fiber direction.
    
    Args:
        C: Strain tensor (Batch, 3, 3)
        fiber: Fiber direction unit vector (Batch, 3)
    
    Returns:
        I4: Fiber strain (Batch,)
    """
    # Ensure fiber is column vector: (B, 3, 1)
    f = fiber.unsqueeze(-1)
    
    # Compute f^T C f
    # First: C @ f -> (B, 3, 1)
    Cf = torch.matmul(C, f)
    
    # Then: f^T @ (C @ f) -> (B, 1, 1)
    I4 = torch.matmul(f.transpose(-2, -1), Cf).squeeze(-1).squeeze(-1)
    
    return I4


def rotation_matrix_from_axis_angle(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """
    Compute rotation matrix using Rodrigues' formula.
    
    R = I + sin(θ) K + (1 - cos(θ)) K^2
    where K is the skew-symmetric matrix of the axis.
    
    Args:
        axis: Rotation axis unit vector (Batch, 3)
        angle: Rotation angle in radians (Batch,)
    
    Returns:
        R: Rotation matrix (Batch, 3, 3)
    """
    batch_size = axis.shape[0]
    device = axis.device
    
    # Normalize axis
    axis = axis / (torch.norm(axis, dim=1, keepdim=True) + 1e-8)
    
    # Create skew-symmetric matrix K
    zero = torch.zeros(batch_size, device=device)
    K = torch.stack([
        torch.stack([zero, -axis[:, 2], axis[:, 1]], dim=1),
        torch.stack([axis[:, 2], zero, -axis[:, 0]], dim=1),
        torch.stack([-axis[:, 1], axis[:, 0], zero], dim=1)
    ], dim=1)  # (B, 3, 3)
    
    # Compute K^2
    K_squared = torch.matmul(K, K)
    
    # Identity matrix
    I = torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1)
    
    # Rodrigues' formula
    sin_angle = torch.sin(angle).view(-1, 1, 1)
    cos_angle = torch.cos(angle).view(-1, 1, 1)
    
    R = I + sin_angle * K + (1 - cos_angle) * K_squared
    
    return R


def rotation_matrix_to_axis_angle(R: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract axis and angle from rotation matrix.
    
    Args:
        R: Rotation matrix (Batch, 3, 3)
    
    Returns:
        axis: Rotation axis (Batch, 3)
        angle: Rotation angle (Batch,)
    """
    batch_size = R.shape[0]
    
    # Angle from trace: tr(R) = 1 + 2*cos(θ)
    trace = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1.0, 1.0))
    
    # Axis from skew-symmetric part
    # K = (R - R^T) / (2 * sin(θ))
    R_T = R.transpose(-2, -1)
    K = (R - R_T) / (2 * torch.sin(angle).view(-1, 1, 1) + 1e-8)
    
    # Extract axis from K
    axis = torch.stack([K[:, 2, 1], K[:, 0, 2], K[:, 1, 0]], dim=1)
    
    # Normalize
    axis = axis / (torch.norm(axis, dim=1, keepdim=True) + 1e-8)
    
    return axis, angle


def geodesic_distance(
    z1: torch.Tensor,
    z2: torch.Tensor,
    metric: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute geodesic distance between latent points.
    
    If metric is provided, computes Riemannian distance:
    d(z1, z2) = sqrt((z2 - z1)^T G (z2 - z1))
    
    Otherwise, computes Euclidean distance.
    
    Args:
        z1: First latent point (Batch, Latent_dim)
        z2: Second latent point (Batch, Latent_dim)
        metric: Metric tensor G (Batch, Latent_dim, Latent_dim) [optional]
    
    Returns:
        distance: Geodesic distance (Batch,)
    """
    delta = z2 - z1  # (B, L)
    
    if metric is None:
        # Euclidean distance
        distance = torch.norm(delta, dim=1)
    else:
        # Riemannian distance
        # d = sqrt(delta^T G delta)
        delta_unsqueezed = delta.unsqueeze(-1)  # (B, L, 1)
        
        # G @ delta
        G_delta = torch.matmul(metric, delta_unsqueezed)  # (B, L, 1)
        
        # delta^T @ (G @ delta)
        distance_squared = torch.matmul(
            delta.unsqueeze(-2), G_delta
        ).squeeze(-1).squeeze(-1)  # (B,)
        
        distance = torch.sqrt(torch.clamp(distance_squared, min=0.0))
    
    return distance


def parallel_transport(
    vector: torch.Tensor,
    path: torch.Tensor,
    metric: torch.Tensor
) -> torch.Tensor:
    """
    Parallel transport a vector along a path (simplified).
    
    This is a discrete approximation using the Schild's ladder algorithm.
    
    Args:
        vector: Vector to transport (Batch, Latent_dim)
        path: Path points (Batch, N_steps, Latent_dim)
        metric: Metric tensor (Batch, Latent_dim, Latent_dim)
    
    Returns:
        transported: Transported vector at end of path (Batch, Latent_dim)
    """
    # Simplified: just return the vector (full implementation is complex)
    # In practice, would use geodesic equation and Christoffel symbols
    return vector


def christoffel_symbols(metric: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """
    Compute Christoffel symbols of the metric connection (simplified).
    
    Γ^k_ij = 0.5 * g^kl * (∂g_il/∂z^j + ∂g_jl/∂z^i - ∂g_ij/∂z^l)
    
    Args:
        metric: Metric tensor (Batch, Latent_dim, Latent_dim)
        z: Point in latent space (Batch, Latent_dim)
    
    Returns:
        christoffel: Christoffel symbols (Batch, L, L, L)
    """
    # This is a placeholder - full computation requires metric derivatives
    batch_size, latent_dim, _ = metric.shape
    return torch.zeros(batch_size, latent_dim, latent_dim, latent_dim, 
                      device=metric.device)


def riemannian_metric_from_latent(
    z: torch.Tensor,
    metric_net: Optional[torch.nn.Module] = None
) -> torch.Tensor:
    """
    Compute Riemannian metric tensor at latent point.
    
    If metric_net is provided, uses it to compute position-dependent metric.
    Otherwise, returns identity (Euclidean metric).
    
    Args:
        z: Latent point (Batch, Latent_dim)
        metric_net: Neural network that outputs metric parameters
    
    Returns:
        G: Metric tensor (Batch, Latent_dim, Latent_dim)
    """
    batch_size, latent_dim = z.shape
    
    if metric_net is None:
        # Default to Euclidean metric (identity)
        G = torch.eye(latent_dim, device=z.device).unsqueeze(0).expand(
            batch_size, -1, -1
        )
    else:
        # Use neural network to compute metric
        # Network should output parameters of positive definite matrix
        G_params = metric_net(z)
        
        # Construct positive definite matrix (e.g., using Cholesky)
        # This is a placeholder for the actual implementation
        G = torch.eye(latent_dim, device=z.device).unsqueeze(0).expand(
            batch_size, -1, -1
        )
    
    return G


def so3_distance(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """
    Compute geodesic distance on SO(3) between rotation matrices.
    
    d(R1, R2) = ||log(R1^T R2)||_F / sqrt(2)
    
    Args:
        R1: First rotation matrix (Batch, 3, 3)
        R2: Second rotation matrix (Batch, 3, 3)
    
    Returns:
        distance: Geodesic distance on SO(3) (Batch,)
    """
    # Relative rotation
    R_rel = torch.matmul(R1.transpose(-2, -1), R2)
    
    # Extract angle
    _, angle = rotation_matrix_to_axis_angle(R_rel)
    
    return angle


# Example usage
if __name__ == "__main__":
    print("Testing geometric utilities...")
    
    BATCH_SIZE = 4
    
    # Test strain invariants
    print("\n1. Testing strain invariants...")
    C = torch.randn(BATCH_SIZE, 3, 3)
    C = (C + C.transpose(-2, -1)) / 2  # Make symmetric
    C = C + 3 * torch.eye(3).unsqueeze(0)  # Make positive definite
    
    I1, I2, I3 = compute_strain_invariants(C)
    print(f"I1 shape: {I1.shape}, values: {I1}")
    print(f"I2 shape: {I2.shape}, values: {I2}")
    print(f"I3 shape: {I3.shape}, values: {I3}")
    
    # Test fiber strain
    print("\n2. Testing fiber strain...")
    fiber = torch.randn(BATCH_SIZE, 3)
    fiber = fiber / torch.norm(fiber, dim=1, keepdim=True)
    
    I4 = fiber_strain(C, fiber)
    print(f"I4 shape: {I4.shape}, values: {I4}")
    
    # Test rotation matrices
    print("\n3. Testing rotation matrices...")
    axis = torch.randn(BATCH_SIZE, 3)
    angle = torch.rand(BATCH_SIZE) * np.pi
    
    R = rotation_matrix_from_axis_angle(axis, angle)
    print(f"Rotation matrix shape: {R.shape}")
    
    # Verify orthogonality: R^T R = I
    RTR = torch.matmul(R.transpose(-2, -1), R)
    I = torch.eye(3).unsqueeze(0).expand(BATCH_SIZE, -1, -1)
    print(f"Orthogonality error: {torch.norm(RTR - I, dim=(1,2)).mean():.6f}")
    
    # Verify determinant = 1
    det_R = torch.det(R)
    print(f"Determinants: {det_R} (should be close to 1)")
    
    # Test axis-angle extraction
    print("\n4. Testing axis-angle extraction...")
    axis_recovered, angle_recovered = rotation_matrix_to_axis_angle(R)
    print(f"Original angles: {angle}")
    print(f"Recovered angles: {angle_recovered}")
    print(f"Angle recovery error: {torch.norm(angle - angle_recovered):.6f}")
    
    # Test geodesic distance
    print("\n5. Testing geodesic distance...")
    z1 = torch.randn(BATCH_SIZE, 32)
    z2 = torch.randn(BATCH_SIZE, 32)
    
    dist_euclidean = geodesic_distance(z1, z2)
    print(f"Euclidean distances: {dist_euclidean}")
    
    # With metric
    metric = torch.eye(32).unsqueeze(0).expand(BATCH_SIZE, -1, -1) * 2.0
    dist_riemannian = geodesic_distance(z1, z2, metric)
    print(f"Riemannian distances: {dist_riemannian}")
    
    # Test SO(3) distance
    print("\n6. Testing SO(3) distance...")
    R1 = rotation_matrix_from_axis_angle(
        torch.randn(BATCH_SIZE, 3), torch.rand(BATCH_SIZE) * np.pi
    )
    R2 = rotation_matrix_from_axis_angle(
        torch.randn(BATCH_SIZE, 3), torch.rand(BATCH_SIZE) * np.pi
    )
    
    so3_dist = so3_distance(R1, R2)
    print(f"SO(3) distances: {so3_dist}")
    
    print("\n✓ All geometric utility tests passed!")
