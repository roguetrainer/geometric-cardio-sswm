"""
SSWM Decoder: Reconstructs observations from latent predictions.

The decoder maps latent representations back to observation space,
generating predicted point clouds with geometric features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSWMDecoder(nn.Module):
    """
    Decoder for Self-Supervised World Model.
    
    Maps predicted latent state z_t+1 back to observation space O_t+1.
    Generates point clouds with positions and features.
    
    Architecture:
        z_t+1 → MLP → Reshape (Folding) → Point Cloud
        
    Args:
        latent_dim (int): Dimension of latent state (default: 32)
        n_points (int): Number of points in output cloud (default: 50)
        output_features (int): Features per point (default: 6)
            - 3 for position (x, y, z)
            - 3 for additional features (fiber orientation, strain, etc.)
        hidden_dim (int): Hidden layer dimension (default: 128)
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        n_points: int = 50,
        output_features: int = 6,
        hidden_dim: int = 128
    ):
        super(SSWMDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.n_points = n_points
        self.output_features = output_features
        self.hidden_dim = hidden_dim
        self.output_dim = n_points * output_features
        
        # Decoder MLP: Expands latent vector to high-dimensional output
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, self.output_dim)
        )
        
    def forward(self, z_t1: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.
        
        Args:
            z_t1: Predicted latent state (Batch, Latent_dim)
        
        Returns:
            O_t1_pred: Predicted observation (Batch, N_points, Output_features)
        """
        batch_size = z_t1.shape[0]
        
        # Step 1: Map latent vector to high-dimensional output
        x = self.mlp(z_t1)  # (B, N*F)
        
        # Step 2: Reshape to point cloud format (folding operation)
        O_t1_pred = x.view(batch_size, self.n_points, self.output_features)
        
        return O_t1_pred


class SSWMDecoderWithStructure(nn.Module):
    """
    Enhanced decoder with explicit geometric structure.
    
    This variant separately decodes different geometric components
    (positions, fiber orientations, features) and applies appropriate
    constraints and normalizations.
    
    Args:
        latent_dim (int): Dimension of latent state
        n_points (int): Number of points in output
        hidden_dim (int): Hidden layer dimension
        normalize_fibers (bool): Whether to normalize fiber orientations
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        n_points: int = 50,
        hidden_dim: int = 128,
        normalize_fibers: bool = True
    ):
        super(SSWMDecoderWithStructure, self).__init__()
        
        self.latent_dim = latent_dim
        self.n_points = n_points
        self.hidden_dim = hidden_dim
        self.normalize_fibers = normalize_fibers
        
        # Shared feature extraction from latent
        self.shared_mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
        )
        
        # Position decoder (x, y, z coordinates)
        self.position_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, n_points * 3)
        )
        
        # Fiber orientation decoder (fiber directions)
        self.fiber_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, n_points * 3)
        )
        
        # Feature decoder (strain invariants, etc.)
        self.feature_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_points * 3)  # 3 additional features
        )
    
    def forward(self, z_t1: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with structured decoding.
        
        Args:
            z_t1: Predicted latent state (Batch, Latent_dim)
        
        Returns:
            O_t1_pred: Predicted observation (Batch, N_points, 9)
                Components: [positions (3), fibers (3), features (3)]
        """
        batch_size = z_t1.shape[0]
        
        # Shared feature extraction
        shared_features = self.shared_mlp(z_t1)  # (B, H*2)
        
        # Decode positions
        positions = self.position_decoder(shared_features)  # (B, N*3)
        positions = positions.view(batch_size, self.n_points, 3)
        
        # Decode fiber orientations
        fibers = self.fiber_decoder(shared_features)  # (B, N*3)
        fibers = fibers.view(batch_size, self.n_points, 3)
        
        # Normalize fiber orientations to unit length
        if self.normalize_fibers:
            fibers = F.normalize(fibers, p=2, dim=2)
        
        # Decode additional features
        features = self.feature_decoder(shared_features)  # (B, N*3)
        features = features.view(batch_size, self.n_points, 3)
        
        # Concatenate all components
        O_t1_pred = torch.cat([positions, fibers, features], dim=2)  # (B, N, 9)
        
        return O_t1_pred


class SSWMDecoderWithGridFolding(nn.Module):
    """
    Decoder using grid folding for structured point cloud generation.
    
    This approach creates a 2D grid which is then "folded" into 3D space,
    useful for maintaining local coherence in the point cloud.
    
    Args:
        latent_dim (int): Dimension of latent state
        grid_size (int): Size of the 2D grid (total points = grid_size^2)
        output_features (int): Features per point
        hidden_dim (int): Hidden layer dimension
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        grid_size: int = 8,  # 8x8 = 64 points
        output_features: int = 6,
        hidden_dim: int = 128
    ):
        super(SSWMDecoderWithGridFolding, self).__init__()
        
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        self.n_points = grid_size * grid_size
        self.output_features = output_features
        self.hidden_dim = hidden_dim
        
        # Create 2D grid coordinates
        # Grid is in range [-1, 1] x [-1, 1]
        grid_coords = torch.meshgrid(
            torch.linspace(-1, 1, grid_size),
            torch.linspace(-1, 1, grid_size),
            indexing='ij'
        )
        grid = torch.stack(grid_coords, dim=-1).reshape(-1, 2)  # (N, 2)
        self.register_buffer('grid', grid)
        
        # Folding network: Maps (latent + grid_coords) to 3D positions + features
        self.fold_mlp = nn.Sequential(
            nn.Linear(latent_dim + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_features)
        )
    
    def forward(self, z_t1: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using grid folding.
        
        Args:
            z_t1: Predicted latent state (Batch, Latent_dim)
        
        Returns:
            O_t1_pred: Predicted observation (Batch, N_points, Output_features)
        """
        batch_size = z_t1.shape[0]
        
        # Expand latent vector to each grid point
        z_expanded = z_t1.unsqueeze(1).expand(-1, self.n_points, -1)  # (B, N, L)
        
        # Expand grid to batch
        grid_expanded = self.grid.unsqueeze(0).expand(batch_size, -1, -1)  # (B, N, 2)
        
        # Concatenate latent and grid coordinates
        combined = torch.cat([z_expanded, grid_expanded], dim=2)  # (B, N, L+2)
        
        # Apply folding network to each point
        O_t1_pred = self.fold_mlp(combined)  # (B, N, F)
        
        return O_t1_pred


# Example usage and testing
if __name__ == "__main__":
    print("Testing SSWMDecoder...")
    
    BATCH_SIZE = 4
    LATENT_DIM = 32
    N_POINTS = 50
    OUTPUT_FEATURES = 6
    
    # Create dummy input
    dummy_z_t1 = torch.randn(BATCH_SIZE, LATENT_DIM)
    
    # Initialize decoder
    decoder = SSWMDecoder(
        latent_dim=LATENT_DIM,
        n_points=N_POINTS,
        output_features=OUTPUT_FEATURES,
        hidden_dim=128
    )
    
    # Forward pass
    O_t1_pred = decoder(dummy_z_t1)
    
    print(f"Input z_t+1 shape: {dummy_z_t1.shape}")
    print(f"Output O_t+1 shape: {O_t1_pred.shape}")
    print(f"Expected shape: ({BATCH_SIZE}, {N_POINTS}, {OUTPUT_FEATURES})")
    assert O_t1_pred.shape == (BATCH_SIZE, N_POINTS, OUTPUT_FEATURES), "Output shape mismatch!"
    print("✓ Basic decoder test passed!")
    
    # Test structured decoder
    print("\nTesting SSWMDecoderWithStructure...")
    
    decoder_struct = SSWMDecoderWithStructure(
        latent_dim=LATENT_DIM,
        n_points=N_POINTS,
        hidden_dim=128,
        normalize_fibers=True
    )
    
    O_t1_pred_struct = decoder_struct(dummy_z_t1)
    
    print(f"Structured output shape: {O_t1_pred_struct.shape}")
    
    # Check fiber normalization
    fibers = O_t1_pred_struct[:, :, 3:6]  # Extract fiber component
    fiber_norms = torch.norm(fibers, p=2, dim=2)
    print(f"Fiber norm range: [{fiber_norms.min():.4f}, {fiber_norms.max():.4f}]")
    assert torch.allclose(fiber_norms, torch.ones_like(fiber_norms), atol=1e-5), \
        "Fibers not properly normalized!"
    print("✓ Structured decoder test passed!")
    
    # Test grid folding decoder
    print("\nTesting SSWMDecoderWithGridFolding...")
    
    GRID_SIZE = 7  # 7x7 = 49 points (close to 50)
    
    decoder_fold = SSWMDecoderWithGridFolding(
        latent_dim=LATENT_DIM,
        grid_size=GRID_SIZE,
        output_features=OUTPUT_FEATURES,
        hidden_dim=128
    )
    
    O_t1_pred_fold = decoder_fold(dummy_z_t1)
    
    print(f"Grid folding output shape: {O_t1_pred_fold.shape}")
    expected_points = GRID_SIZE * GRID_SIZE
    assert O_t1_pred_fold.shape == (BATCH_SIZE, expected_points, OUTPUT_FEATURES), \
        "Grid folding output shape mismatch!"
    print("✓ Grid folding decoder test passed!")
    
    print("\nAll tests passed successfully!")
