import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

# --- Re-use the SynthCardioSSWM class definition and helper functions ---
# (Assuming SSWMEncoder, SSWMDynamicsModel, SSWMDecoder, and SynthCardioSSWM
# classes from previous steps are accessible)

def total_sswm_loss(O_t1_pred, z_t_mu, z_t_logvar, O_t1_true, kl_weight: float = 0.001):
    """
    Calculates the combined Reconstruction Loss and KL Divergence Loss.
    
    Args:
        O_t1_pred: Predicted next observation (Batch, N, 6).
        z_t_mu: Mean of the latent state distribution (Batch, Latent_dim).
        z_t_logvar: Log variance of the latent state distribution (Batch, Latent_dim).
        O_t1_true: Ground-truth next observation (Batch, N, 6).
        kl_weight: Scaling factor for the KL divergence term.
        
    Returns:
        total_loss, rec_loss, kl_loss (scalar tensors).
    """
    
    # 1. Reconstruction Loss (MSE)
    # The SSWM must reconstruct the predicted point cloud and its 6 features (x,y,z, F1, F2, F3)
    rec_loss = F.mse_loss(O_t1_pred, O_t1_true, reduction='mean')
    
    # 2. KL Divergence Loss
    # Formula for KL divergence between N(mu, sigma^2) and N(0, 1):
    # 0.5 * sum(exp(logvar) + mu^2 - 1 - logvar)
    kl_loss = -0.5 * torch.sum(1 + z_t_logvar - z_t_mu.pow(2) - z_t_logvar.exp())
    # Normalize KL loss by batch size or total elements
    kl_loss /= z_t_mu.numel() 
    
    # 3. Total Loss
    total_loss = rec_loss + kl_weight * kl_loss
    
    return total_loss, rec_loss, kl_loss

def train_sswm_step(model: SynthCardioSSWM, batch: list, optimizer: optim.Optimizer, kl_weight: float = 0.001) -> dict:
    """
    Performs one full training step over an entire sequence in a batch.
    """
    model.train()
    optimizer.zero_grad()
    
    # Batch is a list of sequence steps (O_t, A_t, O_t+1)
    
    # Initialize the previous latent state (for the GRU's hidden state)
    z_prev = None 
    total_batch_loss = 0.0
    
    # Iterate through all steps in the sequence
    for step in batch:
        # Load data for this step
        O_t = torch.tensor(step['O_t'], dtype=torch.float32).unsqueeze(0)  # (1, N, 5)
        A_t = torch.tensor(step['A_t'], dtype=torch.float32).unsqueeze(0)  # (1, 3, 3)
        O_t1_true = torch.tensor(step['O_t1'], dtype=torch.float32).unsqueeze(0) # (1, N, 6)
        
        # Forward pass: O_t -> z_t -> z_t+1 -> O_t+1_pred
        O_t1_pred, z_t_sampled, mu, logvar = model(O_t, A_t, z_prev=z_prev)
        
        # Update z_prev for the next step in the sequence
        z_prev = z_t_sampled.detach() # Detach to prevent gradient flow through sequence boundary
        
        # Calculate loss for this step
        loss, rec_loss, kl_loss = total_sswm_loss(O_t1_pred, mu, logvar, O_t1_true, kl_weight)
        
        total_batch_loss += loss.item()
        
        # Backward pass (accumulate gradients across time steps - Truncated BPTT is common)
        loss.backward()

    # Optimize weights
    optimizer.step()
    
    # Return average loss per step for logging
    return {'loss': total_batch_loss / len(batch), 
            'rec_loss': rec_loss.item(), 
            'kl_loss': kl_loss.item()}

# --- Example Training Loop Structure ---
# # Assuming the SynthCardioDataSampler class is available and you have generated 'dataset'
# # Example instantiation (use larger N_POINTS and N_SEQUENCES for real training)
# N_POINTS = 50 
# LATENT_DIM = 32
# N_SEQUENCES = 10
# SEQ_LENGTH = 5

# sampler = SynthCardioDataSampler(n_sequences=N_SEQUENCES, seq_length=SEQ_LENGTH, n_points=N_POINTS)
# dataset = sampler.generate_dataset()

# model = SynthCardioSSWM(n_points=N_POINTS, latent_dim=LATENT_DIM)
# optimizer = optim.Adam(model.parameters(), lr=1e-4)

# print(f"\nStarting training for {N_SEQUENCES} sequences...")
# NUM_EPOCHS = 5

# for epoch in range(NUM_EPOCHS):
#     epoch_loss = 0.0
#     for sequence in dataset:
#         # sequence is a list of steps: [{'O_t', 'A_t', 'O_t1'}, ...]
#         metrics = train_sswm_step(model, sequence, optimizer)
#         epoch_loss += metrics['loss']
        
#     avg_epoch_loss = epoch_loss / N_SEQUENCES
#     print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Avg Total Loss: {avg_epoch_loss:.4f} | Final Rec Loss: {metrics['rec_loss']:.4f}")