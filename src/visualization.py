"""
Visualization utilities for SSWM.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch


def plot_point_cloud(positions, fibers=None, title="Point Cloud", figsize=(10, 8)):
    """Plot 3D point cloud with optional fiber orientations."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], 
               c='blue', alpha=0.6, s=20)
    
    # Plot fibers if provided
    if fibers is not None:
        scale = 0.05
        for i in range(0, len(positions), max(1, len(positions)//50)):
            p = positions[i]
            f = fibers[i] * scale
            ax.quiver(p[0], p[1], p[2], f[0], f[1], f[2],
                     color='red', alpha=0.8, arrow_length_ratio=0.3)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_strain_heatmap(features, title="Strain Features"):
    """Plot heatmap of strain features."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    feature_names = ['I1 (Volume)', 'I4 (Fiber Strain)', 'Shear']
    
    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        im = ax.scatter(features[:, 0], features[:, 1], 
                       c=features[:, i], cmap='viridis', s=50)
        ax.set_title(name)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax)
    
    plt.suptitle(title)
    plt.tight_layout()
    return fig


def plot_latent_space(latents, labels=None, title="Latent Space"):
    """Plot 2D projection of latent space."""
    from sklearn.decomposition import PCA
    
    if latents.shape[1] > 2:
        pca = PCA(n_components=2)
        latents_2d = pca.fit_transform(latents)
    else:
        latents_2d = latents
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if labels is not None:
        scatter = ax.scatter(latents_2d[:, 0], latents_2d[:, 1], 
                           c=labels, cmap='tab10', alpha=0.6)
        plt.colorbar(scatter, ax=ax)
    else:
        ax.scatter(latents_2d[:, 0], latents_2d[:, 1], alpha=0.6)
    
    ax.set_xlabel('PC1' if latents.shape[1] > 2 else 'Dim 1')
    ax.set_ylabel('PC2' if latents.shape[1] > 2 else 'Dim 2')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_prediction_comparison(true_obs, pred_obs, idx=0):
    """Compare true and predicted observations."""
    fig = plt.figure(figsize=(15, 5))
    
    # True observation
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(true_obs[idx, :, 0], true_obs[idx, :, 1], true_obs[idx, :, 2],
               c='blue', alpha=0.6)
    ax1.set_title('True Observation')
    
    # Predicted observation
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(pred_obs[idx, :, 0], pred_obs[idx, :, 1], pred_obs[idx, :, 2],
               c='red', alpha=0.6)
    ax2.set_title('Predicted Observation')
    
    # Overlay
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(true_obs[idx, :, 0], true_obs[idx, :, 1], true_obs[idx, :, 2],
               c='blue', alpha=0.4, label='True')
    ax3.scatter(pred_obs[idx, :, 0], pred_obs[idx, :, 1], pred_obs[idx, :, 2],
               c='red', alpha=0.4, label='Predicted')
    ax3.set_title('Overlay')
    ax3.legend()
    
    plt.tight_layout()
    return fig
