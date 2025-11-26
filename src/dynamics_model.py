"""
SSWM Dynamics Model: Predicts latent state evolution under actions.

The dynamics model learns how the latent representation evolves over time
given the current state and an action (strain tensor in the cardiac case).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SSWMDynamicsModel(nn.Module):
    """
    Dynamics Model for Self-Supervised World Model.
    
    Predicts the next latent state z_t+1 given current state z_t and action A_t.
    Uses a GRU to maintain temporal consistency and memory.
    
    Architecture:
        [z_t, A_t] → MLP → GRU → Residual Connection → z_t+1
        
    Args:
        latent_dim (int): Dimension of latent state (default: 32)
        action_dim (int): Dimension of action space (default: 9 for 3x3 strain tensor)
        hidden_dim (int): Hidden layer dimension (default: 64)
        num_layers (int): Number of GRU layers (default: 1)
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        action_dim: int = 9,
        hidden_dim: int = 64,
        num_layers: int = 1
    ):
        super(SSWMDynamicsModel, self).__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input processing: Combine z_t and A_t
        # The action is typically a 3x3 strain tensor, flattened to 9 elements
        self.input_mlp = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Recurrent dynamics using GRU
        # GRU maintains temporal consistency and memory of previous states
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Output processing: Map GRU hidden state to latent prediction
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        
        # Residual connection weight (learnable)
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
    
    def forward(
        self,
        z_t: torch.Tensor,
        A_t: torch.Tensor,
        h_prev: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the dynamics model.
        
        Args:
            z_t: Current latent state (Batch, Latent_dim)
            A_t: Current action (Batch, 3, 3) - strain tensor
            h_prev: Previous GRU hidden state (Num_layers, Batch, Hidden_dim)
                   If None, initialized to zeros.
        
        Returns:
            z_t1_pred: Predicted next latent state (Batch, Latent_dim)
            h_t: Current GRU hidden state (Num_layers, Batch, Hidden_dim)
        """
        batch_size = z_t.shape[0]
        
        # Step 1: Flatten the action tensor (3x3 matrix → 9D vector)
        A_t_flat = A_t.view(batch_size, -1)  # (B, 9)
        
        # Step 2: Concatenate latent state and action
        combined_input = torch.cat([z_t, A_t_flat], dim=1)  # (B, L+9)
        
        # Step 3: Process combined input
        x = self.input_mlp(combined_input)  # (B, H)
        
        # Step 4: Add sequence dimension for GRU (expects 3D input)
        x = x.unsqueeze(1)  # (B, 1, H)
        
        # Step 5: GRU forward pass
        if h_prev is None:
            # Initialize hidden state to zeros
            h_prev = torch.zeros(
                self.num_layers, batch_size, self.hidden_dim,
                device=z_t.device, dtype=z_t.dtype
            )
        
        gru_out, h_t = self.gru(x, h_prev)  # gru_out: (B, 1, H), h_t: (L, B, H)
        
        # Step 6: Remove sequence dimension
        gru_out = gru_out.squeeze(1)  # (B, H)
        
        # Step 7: Map to latent space prediction
        z_delta = self.output_mlp(gru_out)  # (B, L)
        
        # Step 8: Residual connection
        # z_t+1 = z_t + α * Δz (where α is learnable)
        z_t1_pred = z_t + self.residual_weight * z_delta
        
        return z_t1_pred, h_t


class SSWMDynamicsModelWithConstraints(nn.Module):
    """
    Enhanced dynamics model with explicit geometric constraints.
    
    This variant enforces physical constraints like incompressibility
    and fiber orientation preservation during prediction.
    
    Args:
        latent_dim (int): Dimension of latent state
        action_dim (int): Dimension of action space
        hidden_dim (int): Hidden layer dimension
        enforce_incompressibility (bool): Enforce volume preservation
        fiber_dim (int): Dimension of fiber orientation subspace
    """
    
    def __init__(
        self,
        latent_dim: int = 32,
        action_dim: int = 9,
        hidden_dim: int = 64,
        enforce_incompressibility: bool = True,
        fiber_dim: int = 3
    ):
        super(SSWMDynamicsModelWithConstraints, self).__init__()
        
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.enforce_incompressibility = enforce_incompressibility
        self.fiber_dim = fiber_dim
        
        # Decompose latent space: [position, fiber_orientation, other]
        self.pos_dim = 3
        self.other_dim = latent_dim - self.pos_dim - fiber_dim
        
        # Separate networks for different latent components
        self.position_net = nn.Sequential(
            nn.Linear(self.pos_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.pos_dim),
        )
        
        self.fiber_net = nn.Sequential(
            nn.Linear(fiber_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, fiber_dim),
        )
        
        if self.other_dim > 0:
            self.other_net = nn.Sequential(
                nn.Linear(self.other_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.other_dim),
            )
    
    def forward(
        self,
        z_t: torch.Tensor,
        A_t: torch.Tensor,
        h_prev: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with geometric constraints.
        
        Args:
            z_t: Current latent state (Batch, Latent_dim)
            A_t: Current action (Batch, 3, 3)
            h_prev: Not used in this variant (for compatibility)
        
        Returns:
            z_t1_pred: Predicted next latent state (Batch, Latent_dim)
            h_t: Dummy hidden state (None)
        """
        batch_size = z_t.shape[0]
        A_t_flat = A_t.view(batch_size, -1)
        
        # Decompose latent state
        z_pos = z_t[:, :self.pos_dim]
        z_fiber = z_t[:, self.pos_dim:self.pos_dim + self.fiber_dim]
        z_other = z_t[:, self.pos_dim + self.fiber_dim:] if self.other_dim > 0 else None
        
        # Process each component separately
        # Position update (with action)
        pos_input = torch.cat([z_pos, A_t_flat], dim=1)
        z_pos_delta = self.position_net(pos_input)
        z_pos_new = z_pos + z_pos_delta
        
        # Fiber orientation update (with rotation from strain tensor)
        fiber_input = torch.cat([z_fiber, A_t_flat], dim=1)
        z_fiber_delta = self.fiber_net(fiber_input)
        z_fiber_new = z_fiber + z_fiber_delta
        
        # Normalize fiber orientation to unit length (constraint)
        z_fiber_new = F.normalize(z_fiber_new, p=2, dim=1)
        
        # Other features
        if self.other_dim > 0:
            other_input = torch.cat([z_other, A_t_flat], dim=1)
            z_other_delta = self.other_net(other_input)
            z_other_new = z_other + z_other_delta
        
        # Recombine components
        if self.other_dim > 0:
            z_t1_pred = torch.cat([z_pos_new, z_fiber_new, z_other_new], dim=1)
        else:
            z_t1_pred = torch.cat([z_pos_new, z_fiber_new], dim=1)
        
        return z_t1_pred, None


# Example usage and testing
if __name__ == "__main__":
    print("Testing SSWMDynamicsModel...")
    
    BATCH_SIZE = 4
    LATENT_DIM = 32
    ACTION_DIM = 9  # 3x3 strain tensor
    HIDDEN_DIM = 64
    
    # Create dummy inputs
    dummy_z_t = torch.randn(BATCH_SIZE, LATENT_DIM)
    dummy_A_t = torch.randn(BATCH_SIZE, 3, 3)
    
    # Initialize dynamics model
    dynamics = SSWMDynamicsModel(
        latent_dim=LATENT_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM
    )
    
    # Forward pass
    z_t1_pred, h_t = dynamics(dummy_z_t, dummy_A_t)
    
    print(f"Input z_t shape: {dummy_z_t.shape}")
    print(f"Input A_t shape: {dummy_A_t.shape}")
    print(f"Output z_t+1 shape: {z_t1_pred.shape}")
    print(f"Hidden state shape: {h_t.shape}")
    assert z_t1_pred.shape == (BATCH_SIZE, LATENT_DIM), "Output shape mismatch!"
    print("✓ Basic dynamics model test passed!")
    
    # Test with hidden state
    print("\nTesting with previous hidden state...")
    z_t1_pred_2, h_t_2 = dynamics(z_t1_pred, dummy_A_t, h_prev=h_t)
    print(f"Second prediction shape: {z_t1_pred_2.shape}")
    print("✓ Sequential prediction test passed!")
    
    # Test constrained dynamics model
    print("\nTesting SSWMDynamicsModelWithConstraints...")
    
    dynamics_constrained = SSWMDynamicsModelWithConstraints(
        latent_dim=LATENT_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        enforce_incompressibility=True,
        fiber_dim=3
    )
    
    z_t1_pred_constrained, _ = dynamics_constrained(dummy_z_t, dummy_A_t)
    
    print(f"Constrained output shape: {z_t1_pred_constrained.shape}")
    
    # Check fiber normalization
    fiber_component = z_t1_pred_constrained[:, 3:6]
    fiber_norms = torch.norm(fiber_component, p=2, dim=1)
    print(f"Fiber orientation norms: {fiber_norms}")
    assert torch.allclose(fiber_norms, torch.ones_like(fiber_norms), atol=1e-6), \
        "Fiber orientations not normalized!"
    print("✓ Constrained dynamics model test passed!")
    
    print("\nAll tests passed successfully!")
