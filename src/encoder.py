"""
SSWM Encoder: Maps observations (point clouds) to latent representations.

The encoder uses a PointNet-inspired architecture to process permutation-invariant
point cloud data representing cardiac geometry and fiber orientations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSWMEncoder(nn.Module):
    """
    Encoder for Self-Supervised World Model.
    
    Maps point cloud observations O_t to latent representations z_t.
    Uses a PointNet-style architecture for permutation invariance.
    
    Architecture:
        Point-wise MLPs → Max Pooling → Global MLP → Latent vector
        
    Args:
        input_features (int): Number of features per point (default: 5)
            - 3 for position (x, y, z)
            - 2 for fiber orientation (theta, phi in spherical coords)
        latent_dim (int): Dimension of latent space (default: 32)
        hidden_dim (int): Hidden layer dimension (default: 64)
    """
    
    def __init__(
        self,
        input_features: int = 5,
        latent_dim: int = 32,
        hidden_dim: int = 64
    ):
        super(SSWMEncoder, self).__init__()
        
        self.input_features = input_features
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Point-wise feature extraction (shared MLP for each point)
        # This processes each point independently before aggregation
        self.point_mlp = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        
        # Global feature extraction after max pooling
        # This processes the aggregated global representation
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
    
    def forward(self, O_t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.
        
        Args:
            O_t: Point cloud observation (Batch, N_points, Input_features)
                Shape: (B, N, F) where:
                    B = batch size
                    N = number of points
                    F = input_features (position + fiber orientation)
        
        Returns:
            z_t: Latent representation (Batch, Latent_dim)
                Shape: (B, L) where L = latent_dim
        """
        batch_size = O_t.shape[0]
        n_points = O_t.shape[1]
        
        # Step 1: Point-wise feature extraction
        # Process each point independently with shared weights
        # Input: (B, N, F) → Output: (B, N, H*2)
        point_features = self.point_mlp(O_t)
        
        # Step 2: Global aggregation via max pooling
        # Take the maximum value across all points for each feature dimension
        # This creates a permutation-invariant global representation
        # Input: (B, N, H*2) → Output: (B, H*2)
        global_features, _ = torch.max(point_features, dim=1)
        
        # Step 3: Global feature processing
        # Map the aggregated features to the latent space
        # Input: (B, H*2) → Output: (B, L)
        z_t = self.global_mlp(global_features)
        
        return z_t


class SSWMEncoderWithAttention(nn.Module):
    """
    Enhanced encoder with attention mechanism for adaptive point weighting.
    
    This variant uses attention to weight different points differently,
    which can be useful for focusing on regions of high deformation.
    
    Args:
        input_features (int): Number of features per point
        latent_dim (int): Dimension of latent space
        hidden_dim (int): Hidden layer dimension
        num_heads (int): Number of attention heads (default: 4)
    """
    
    def __init__(
        self,
        input_features: int = 5,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        num_heads: int = 4
    ):
        super(SSWMEncoderWithAttention, self).__init__()
        
        self.input_features = input_features
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Point-wise feature extraction
        self.point_mlp = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Multi-head attention for adaptive pooling
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Query vector for attention (learnable)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Global feature extraction
        self.global_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
    
    def forward(self, O_t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with attention-based aggregation.
        
        Args:
            O_t: Point cloud observation (Batch, N_points, Input_features)
        
        Returns:
            z_t: Latent representation (Batch, Latent_dim)
        """
        batch_size = O_t.shape[0]
        
        # Point-wise feature extraction
        point_features = self.point_mlp(O_t)  # (B, N, H)
        
        # Expand query to batch size
        query = self.query.expand(batch_size, -1, -1)  # (B, 1, H)
        
        # Attention-based aggregation
        # The attention mechanism learns to weight different points
        global_features, attention_weights = self.attention(
            query, point_features, point_features
        )  # (B, 1, H)
        
        # Squeeze the sequence dimension
        global_features = global_features.squeeze(1)  # (B, H)
        
        # Global feature processing
        z_t = self.global_mlp(global_features)  # (B, L)
        
        return z_t


# Example usage and testing
if __name__ == "__main__":
    # Test the basic encoder
    print("Testing SSWMEncoder...")
    
    BATCH_SIZE = 4
    N_POINTS = 50
    INPUT_FEATURES = 5  # (x, y, z, theta, phi)
    LATENT_DIM = 32
    
    # Create dummy input
    dummy_observation = torch.randn(BATCH_SIZE, N_POINTS, INPUT_FEATURES)
    
    # Initialize encoder
    encoder = SSWMEncoder(
        input_features=INPUT_FEATURES,
        latent_dim=LATENT_DIM,
        hidden_dim=64
    )
    
    # Forward pass
    latent = encoder(dummy_observation)
    
    print(f"Input shape: {dummy_observation.shape}")
    print(f"Output shape: {latent.shape}")
    print(f"Expected output shape: ({BATCH_SIZE}, {LATENT_DIM})")
    assert latent.shape == (BATCH_SIZE, LATENT_DIM), "Output shape mismatch!"
    print("✓ Basic encoder test passed!")
    
    # Test attention-based encoder
    print("\nTesting SSWMEncoderWithAttention...")
    
    encoder_att = SSWMEncoderWithAttention(
        input_features=INPUT_FEATURES,
        latent_dim=LATENT_DIM,
        hidden_dim=64,
        num_heads=4
    )
    
    latent_att = encoder_att(dummy_observation)
    
    print(f"Input shape: {dummy_observation.shape}")
    print(f"Output shape: {latent_att.shape}")
    assert latent_att.shape == (BATCH_SIZE, LATENT_DIM), "Output shape mismatch!"
    print("✓ Attention encoder test passed!")
    
    print("\nAll tests passed successfully!")
