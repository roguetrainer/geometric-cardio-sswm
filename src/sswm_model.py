"""
Complete SSWM Model: Integrates encoder, dynamics, and decoder.

This module brings together all components into a trainable Self-Supervised
World Model for cardiac mechanics prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .encoder import SSWMEncoder
from .dynamics_model import SSWMDynamicsModel
from .decoder import SSWMDecoder


class SynthCardioSSWM(nn.Module):
    """
    Complete Self-Supervised World Model for Synthetic Cardiac Data.
    
    This model integrates three components:
    1. Encoder: O_t → z_t
    2. Dynamics: (z_t, A_t) → z_t+1
    3. Decoder: z_t+1 → Ô_t+1
    
    The model is trained to predict future observations given current
    observations and actions, learning a compressed latent representation
    that captures the essential dynamics.
    
    Args:
        n_points (int): Number of points in point cloud
        latent_dim (int): Dimension of latent space
        hidden_dim (int): Hidden layer dimension
        action_dim (int): Dimension of action (default: 9 for 3x3 strain tensor)
        input_features (int): Features per input point
        output_features (int): Features per output point
        use_vae (bool): Whether to use VAE regularization
    """
    
    def __init__(
        self,
        n_points: int = 50,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        action_dim: int = 9,
        input_features: int = 5,
        output_features: int = 6,
        use_vae: bool = True
    ):
        super(SynthCardioSSWM, self).__init__()
        
        self.n_points = n_points
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.use_vae = use_vae
        
        # Initialize components
        self.encoder = SSWMEncoder(
            input_features=input_features,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim
        )
        
        self.dynamics_model = SSWMDynamicsModel(
            latent_dim=latent_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim
        )
        
        self.decoder = SSWMDecoder(
            latent_dim=latent_dim,
            n_points=n_points,
            output_features=output_features,
            hidden_dim=hidden_dim * 2
        )
        
        # VAE components (for robust latent space)
        if self.use_vae:
            self.fc_mu = nn.Linear(latent_dim, latent_dim)
            self.fc_logvar = nn.Linear(latent_dim, latent_dim)
    
    def reparameterize(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        Reparameterization trick for VAE.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def encode(
        self,
        O_t: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Encode observation to latent space.
        
        Args:
            O_t: Current observation (Batch, N_points, Input_features)
        
        Returns:
            z_t: Latent representation (Batch, Latent_dim)
            mu: Mean (if VAE) (Batch, Latent_dim)
            logvar: Log variance (if VAE) (Batch, Latent_dim)
        """
        # Encode to pre-latent
        z_t_pre = self.encoder(O_t)
        
        if self.use_vae:
            # VAE path: compute mu and logvar
            mu = self.fc_mu(z_t_pre)
            logvar = self.fc_logvar(z_t_pre)
            z_t = self.reparameterize(mu, logvar)
            return z_t, mu, logvar
        else:
            # Direct encoding (no VAE)
            return z_t_pre, None, None
    
    def predict_next_latent(
        self,
        z_t: torch.Tensor,
        A_t: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict next latent state using dynamics model.
        
        Args:
            z_t: Current latent state (Batch, Latent_dim)
            A_t: Action (Batch, 3, 3)
            h_prev: Previous hidden state (optional)
        
        Returns:
            z_t1: Predicted next latent state (Batch, Latent_dim)
            h_t: Current hidden state
        """
        z_t1, h_t = self.dynamics_model(z_t, A_t, h_prev)
        return z_t1, h_t
    
    def decode(
        self,
        z_t1: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode latent state to observation space.
        
        Args:
            z_t1: Latent state (Batch, Latent_dim)
        
        Returns:
            O_t1_pred: Predicted observation (Batch, N_points, Output_features)
        """
        return self.decoder(z_t1)
    
    def forward(
        self,
        O_t: torch.Tensor,
        A_t: torch.Tensor,
        h_prev: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Full forward pass: O_t → z_t → z_t+1 → Ô_t+1
        
        Args:
            O_t: Current observation (Batch, N_points, Input_features)
            A_t: Action (Batch, 3, 3)
            h_prev: Previous hidden state (optional)
        
        Returns:
            O_t1_pred: Predicted next observation (Batch, N_points, Output_features)
            z_t: Current latent state (Batch, Latent_dim)
            mu: Mean (if VAE)
            logvar: Log variance (if VAE)
        """
        # Step 1: Encode current observation
        z_t, mu, logvar = self.encode(O_t)
        
        # Step 2: Predict next latent state
        z_t1, h_t = self.predict_next_latent(z_t, A_t, h_prev)
        
        # Step 3: Decode predicted latent state
        O_t1_pred = self.decode(z_t1)
        
        return O_t1_pred, z_t, mu, logvar
    
    def rollout(
        self,
        O_0: torch.Tensor,
        actions: torch.Tensor,
        n_steps: Optional[int] = None
    ) -> torch.Tensor:
        """
        Multi-step prediction rollout.
        
        Args:
            O_0: Initial observation (Batch, N_points, Input_features)
            actions: Sequence of actions (Batch, N_steps, 3, 3)
            n_steps: Number of steps (if None, use actions.shape[1])
        
        Returns:
            predictions: Sequence of predictions (Batch, N_steps, N_points, Output_features)
        """
        if n_steps is None:
            n_steps = actions.shape[1]
        
        batch_size = O_0.shape[0]
        predictions = []
        
        # Initialize
        O_t = O_0
        h_prev = None
        
        for t in range(n_steps):
            # Get action for this timestep
            A_t = actions[:, t]
            
            # Predict next observation
            O_t1_pred, z_t, _, _ = self.forward(O_t, A_t, h_prev)
            
            predictions.append(O_t1_pred)
            
            # Update for next iteration
            # Note: We use predicted observation as input (teacher forcing can be added)
            O_t = O_t1_pred
            h_prev = z_t.unsqueeze(0)  # Use current latent as hidden state
        
        # Stack predictions
        predictions = torch.stack(predictions, dim=1)  # (B, T, N, F)
        
        return predictions
    
    def compute_loss(
        self,
        O_t1_pred: torch.Tensor,
        O_t1_true: torch.Tensor,
        mu: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None,
        beta: float = 0.001
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute training loss.
        
        Args:
            O_t1_pred: Predicted observation (Batch, N, F)
            O_t1_true: True observation (Batch, N, F)
            mu: Mean from VAE (if applicable)
            logvar: Log variance from VAE (if applicable)
            beta: Weight for KL divergence loss
        
        Returns:
            total_loss: Combined loss
            loss_dict: Dictionary of individual loss components
        """
        # Reconstruction loss (MSE)
        loss_recon = F.mse_loss(O_t1_pred, O_t1_true)
        
        loss_dict = {'reconstruction': loss_recon.item()}
        total_loss = loss_recon
        
        # KL divergence loss (if VAE)
        if self.use_vae and mu is not None and logvar is not None:
            # KL(N(mu, sigma) || N(0, 1))
            loss_kl = -0.5 * torch.mean(
                1 + logvar - mu.pow(2) - logvar.exp()
            )
            loss_dict['kl_divergence'] = loss_kl.item()
            total_loss = total_loss + beta * loss_kl
        
        loss_dict['total'] = total_loss.item()
        
        return total_loss, loss_dict


# Example usage and testing
if __name__ == "__main__":
    print("Testing SynthCardioSSWM...")
    
    BATCH_SIZE = 4
    N_POINTS = 50
    INPUT_FEATURES = 5  # (x, y, z, theta, phi)
    OUTPUT_FEATURES = 6  # (x, y, z, f1, f2, f3)
    LATENT_DIM = 32
    HIDDEN_DIM = 64
    
    # Create dummy data
    dummy_O_t = torch.randn(BATCH_SIZE, N_POINTS, INPUT_FEATURES)
    dummy_A_t = torch.randn(BATCH_SIZE, 3, 3)
    dummy_O_t1_true = torch.randn(BATCH_SIZE, N_POINTS, OUTPUT_FEATURES)
    
    # Initialize model
    model = SynthCardioSSWM(
        n_points=N_POINTS,
        latent_dim=LATENT_DIM,
        hidden_dim=HIDDEN_DIM,
        input_features=INPUT_FEATURES,
        output_features=OUTPUT_FEATURES,
        use_vae=True
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    O_t1_pred, z_t, mu, logvar = model(dummy_O_t, dummy_A_t)
    
    print(f"Input O_t shape: {dummy_O_t.shape}")
    print(f"Action A_t shape: {dummy_A_t.shape}")
    print(f"Predicted O_t+1 shape: {O_t1_pred.shape}")
    print(f"Latent z_t shape: {z_t.shape}")
    if mu is not None:
        print(f"VAE mu shape: {mu.shape}")
        print(f"VAE logvar shape: {logvar.shape}")
    
    assert O_t1_pred.shape == (BATCH_SIZE, N_POINTS, OUTPUT_FEATURES), "Prediction shape mismatch!"
    print("✓ Forward pass test passed!")
    
    # Test loss computation
    print("\nTesting loss computation...")
    total_loss, loss_dict = model.compute_loss(
        O_t1_pred, dummy_O_t1_true, mu, logvar, beta=0.001
    )
    
    print(f"Total loss: {total_loss.item():.6f}")
    print(f"Loss components: {loss_dict}")
    print("✓ Loss computation test passed!")
    
    # Test multi-step rollout
    print("\nTesting multi-step rollout...")
    N_STEPS = 5
    dummy_actions = torch.randn(BATCH_SIZE, N_STEPS, 3, 3)
    
    predictions = model.rollout(dummy_O_t, dummy_actions)
    
    print(f"Rollout input shape: {dummy_O_t.shape}")
    print(f"Actions shape: {dummy_actions.shape}")
    print(f"Rollout predictions shape: {predictions.shape}")
    assert predictions.shape == (BATCH_SIZE, N_STEPS, N_POINTS, OUTPUT_FEATURES), \
        "Rollout shape mismatch!"
    print("✓ Rollout test passed!")
    
    # Test backward pass
    print("\nTesting backward pass...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    optimizer.zero_grad()
    O_t1_pred, z_t, mu, logvar = model(dummy_O_t, dummy_A_t)
    loss, _ = model.compute_loss(O_t1_pred, dummy_O_t1_true, mu, logvar)
    loss.backward()
    optimizer.step()
    
    print("✓ Backward pass test passed!")
    
    print("\nAll tests passed successfully!")
