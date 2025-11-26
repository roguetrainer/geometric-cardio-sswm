"""
Training utilities for SSWM model.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from typing import Optional, Dict
import matplotlib.pyplot as plt


def train_sswm(
    model: nn.Module,
    train_data: dict,
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    beta: float = 0.001,
    device: str = 'cpu',
    verbose: bool = True
) -> Dict[str, list]:
    """
    Train the SSWM model.
    
    Args:
        model: SynthCardioSSWM model
        train_data: Training data dictionary
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        beta: KL divergence weight
        device: Device to train on
        verbose: Whether to print progress
    
    Returns:
        history: Dictionary of training metrics
    """
    model = model.to(device)
    model.train()
    
    # Prepare data
    O_t = torch.from_numpy(train_data['observations_input']).float()
    O_t1 = torch.from_numpy(train_data['observations_output']).float()
    A_t = torch.from_numpy(train_data['strain_tensors']).float()
    
    dataset = TensorDataset(O_t, A_t, O_t1)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {'loss': [], 'recon_loss': [], 'kl_loss': []}
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        
        for batch_O_t, batch_A_t, batch_O_t1 in dataloader:
            batch_O_t = batch_O_t.to(device)
            batch_A_t = batch_A_t.to(device)
            batch_O_t1 = batch_O_t1.to(device)
            
            # Forward pass
            O_t1_pred, z_t, mu, logvar = model(batch_O_t, batch_A_t)
            
            # Compute loss
            loss, loss_dict = model.compute_loss(
                O_t1_pred, batch_O_t1, mu, logvar, beta=beta
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            epoch_loss += loss.item()
            epoch_recon += loss_dict['reconstruction']
            if 'kl_divergence' in loss_dict:
                epoch_kl += loss_dict['kl_divergence']
        
        # Average over batches
        n_batches = len(dataloader)
        epoch_loss /= n_batches
        epoch_recon /= n_batches
        epoch_kl /= n_batches
        
        # Store history
        history['loss'].append(epoch_loss)
        history['recon_loss'].append(epoch_recon)
        history['kl_loss'].append(epoch_kl)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.6f} "
                  f"(Recon: {epoch_recon:.6f}, KL: {epoch_kl:.6f})")
    
    return history


def plot_training_history(history: Dict[str, list]):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history['loss'])
    axes[0].set_title('Total Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    
    axes[1].plot(history['recon_loss'], label='Reconstruction')
    axes[1].plot(history['kl_loss'], label='KL Divergence')
    axes[1].set_title('Loss Components')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    return fig
