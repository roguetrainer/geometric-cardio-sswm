import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm # Colormap utilities

# --- Import the necessary classes/functions from the data generation code ---
# NOTE: To run this successfully, you must have the EllipsoidGenerator class
# and get_ellipsoid_surface/calculate_synthetic_features functions defined 
# and accessible in your environment (as they were defined in Step 1).

# Using constants defined earlier:
A_BASE, B_BASE, C_BASE = 5.0, 3.0, 3.0

def visualize_synthcardio_ot(points: np.ndarray, features: np.ndarray, title: str):
    """
    Visualizes the SynthCardio point cloud (O_t) color-coded by the Fiber Angle (F1).
    
    Args:
        points: (N, 3) array of (x, y, z) coordinates.
        features: (N, 2) array of (F1, F2) features.
        title: Title for the plot.
    """
    # Extract the fiber angle (F1) feature for color coding
    # F1 is the first column of the features array (index 0)
    fiber_angle_f1 = features[:, 0] 
    
    # Convert points and colors to lists for 3D plotting functions
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    
    # --- Start Plotting ---
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Use 'viridis' colormap to represent the fiber angle (e.g., -60deg to +60deg)
    scatter = ax.scatter(x, y, z, 
                         c=fiber_angle_f1, 
                         cmap=cm.hsv,  # Use a cyclical colormap (e.g., hsv) for angles
                         s=5)          # Point size
    
    # --- Formatting and Labels ---
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('X (A-axis)')
    ax.set_ylabel('Y (B-axis)')
    ax.set_zlabel('Z (C-axis)')
    
    # Add a color bar to show the range of the fiber angle (F1)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label('Synthetic Myocardial Fiber Angle (Radians)', rotation=270, labelpad=20)
    
    # Set equal aspect ratio to ensure the ellipsoid looks correctly shaped
    ax.set_box_aspect([np.ptp(x), np.ptp(y), np.ptp(z)]) # Set aspect ratio
    
    # Optionally rotate the view for a better perspective
    ax.view_init(elev=20, azim=45) 

    plt.show()

# --- Example Execution (requires EllipsoidGenerator to be defined) ---
# Assuming EllipsoidGenerator and related functions are defined from Step 1:
# generator = EllipsoidGenerator(n_points=1000)
# points_t, features_t = generator.generate_observation(A_BASE, B_BASE, C_BASE)
# visualize_synthcardio_ot(points_t, features_t, 
#                          title=f'SynthCardio Observation O_t (Baseline Ellipsoid)\nColor-Coded by Fiber Angle (F1)')