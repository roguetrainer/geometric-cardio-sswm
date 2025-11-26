"""
Synthetic Cardiac Data Generator

Generates synthetic point cloud data representing cardiac geometry
with fiber orientations and strain patterns.
"""

import numpy as np
import torch
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class CardiacGeometry:
    """Container for cardiac geometry parameters."""
    n_points: int = 100
    radius_endo: float = 0.4  # Inner radius (endocardium)
    radius_epi: float = 0.6   # Outer radius (epicardium)
    height: float = 1.0       # Ventricular height
    fiber_angle_endo: float = -60.0  # Fiber angle at endocardium (degrees)
    fiber_angle_epi: float = 60.0    # Fiber angle at epicardium (degrees)


class SynthCardioDataGenerator:
    """
    Generator for synthetic cardiac geometry data.
    
    Creates point clouds representing the ventricular wall with:
    - 3D positions on a cylindrical/ellipsoidal surface
    - Fiber orientations (helical pattern from endo to epi)
    - Strain tensors representing deformation
    
    Args:
        n_points (int): Number of points to generate
        geometry (CardiacGeometry): Geometric parameters
        noise_level (float): Gaussian noise level for realism
        seed (Optional[int]): Random seed for reproducibility
    """
    
    def __init__(
        self,
        n_points: int = 100,
        geometry: Optional[CardiacGeometry] = None,
        noise_level: float = 0.01,
        seed: Optional[int] = None
    ):
        self.n_points = n_points
        self.geometry = geometry or CardiacGeometry(n_points=n_points)
        self.noise_level = noise_level
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
    
    def generate_cylindrical_points(
        self,
        batch_size: int = 1
    ) -> np.ndarray:
        """
        Generate points on a cylindrical shell (heart wall).
        
        Args:
            batch_size (int): Number of samples to generate
        
        Returns:
            positions: Point positions (Batch, N_points, 3)
        """
        positions = []
        
        for _ in range(batch_size):
            # Sample points uniformly in cylindrical coordinates
            theta = np.random.uniform(0, 2*np.pi, self.n_points)  # Circumferential
            z = np.random.uniform(0, self.geometry.height, self.n_points)  # Longitudinal
            
            # Sample radial position (transmural depth)
            # More points toward mid-wall for better sampling
            transmural_depth = np.random.beta(2, 2, self.n_points)
            r = (self.geometry.radius_endo + 
                 transmural_depth * (self.geometry.radius_epi - self.geometry.radius_endo))
            
            # Convert to Cartesian coordinates
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            
            # Stack into point cloud
            pts = np.stack([x, y, z], axis=1)  # (N, 3)
            
            # Add noise for realism
            pts += np.random.normal(0, self.noise_level, pts.shape)
            
            positions.append(pts)
        
        return np.array(positions)  # (B, N, 3)
    
    def compute_fiber_orientations(
        self,
        positions: np.ndarray
    ) -> np.ndarray:
        """
        Compute fiber orientations following helical pattern.
        
        Fibers rotate from -60° at endocardium to +60° at epicardium.
        
        Args:
            positions: Point positions (Batch, N_points, 3)
        
        Returns:
            fibers: Unit fiber vectors (Batch, N_points, 3)
        """
        batch_size, n_pts, _ = positions.shape
        fibers = np.zeros_like(positions)
        
        for b in range(batch_size):
            for i in range(n_pts):
                x, y, z = positions[b, i]
                
                # Compute radial distance
                r = np.sqrt(x**2 + y**2)
                
                # Compute transmural depth (0 = endo, 1 = epi)
                transmural = (r - self.geometry.radius_endo) / \
                             (self.geometry.radius_epi - self.geometry.radius_endo)
                transmural = np.clip(transmural, 0, 1)
                
                # Fiber angle varies linearly through wall
                fiber_angle = (self.geometry.fiber_angle_endo + 
                              transmural * (self.geometry.fiber_angle_epi - 
                                          self.geometry.fiber_angle_endo))
                fiber_angle_rad = np.radians(fiber_angle)
                
                # Circumferential direction
                theta = np.arctan2(y, x)
                circ_x = -np.sin(theta)
                circ_y = np.cos(theta)
                circ_z = 0
                
                # Longitudinal direction
                long_x = 0
                long_y = 0
                long_z = 1
                
                # Fiber = rotation in circ-long plane
                fiber_x = np.cos(fiber_angle_rad) * circ_x + np.sin(fiber_angle_rad) * long_x
                fiber_y = np.cos(fiber_angle_rad) * circ_y + np.sin(fiber_angle_rad) * long_y
                fiber_z = np.cos(fiber_angle_rad) * circ_z + np.sin(fiber_angle_rad) * long_z
                
                # Normalize
                fiber = np.array([fiber_x, fiber_y, fiber_z])
                fiber = fiber / (np.linalg.norm(fiber) + 1e-8)
                
                fibers[b, i] = fiber
        
        return fibers
    
    def generate_strain_tensor(
        self,
        batch_size: int = 1,
        deformation_type: str = 'contraction'
    ) -> np.ndarray:
        """
        Generate strain tensors representing cardiac deformation.
        
        Args:
            batch_size (int): Number of samples
            deformation_type (str): Type of deformation
                - 'contraction': systolic contraction
                - 'relaxation': diastolic relaxation
                - 'twist': torsional deformation
        
        Returns:
            strain_tensors: (Batch, 3, 3) symmetric positive definite matrices
        """
        tensors = []
        
        for _ in range(batch_size):
            if deformation_type == 'contraction':
                # Systolic: circumferential and longitudinal shortening
                lambda_circ = np.random.uniform(0.7, 0.9)  # Shortening
                lambda_long = np.random.uniform(0.8, 0.95)
                lambda_rad = 1.0 / (lambda_circ * lambda_long)  # Incompressibility
                
            elif deformation_type == 'relaxation':
                # Diastolic: expansion
                lambda_circ = np.random.uniform(1.0, 1.2)
                lambda_long = np.random.uniform(1.0, 1.15)
                lambda_rad = 1.0 / (lambda_circ * lambda_long)
                
            elif deformation_type == 'twist':
                # Torsional with shear
                lambda_circ = np.random.uniform(0.9, 1.1)
                lambda_long = np.random.uniform(0.9, 1.1)
                lambda_rad = 1.0 / (lambda_circ * lambda_long)
                shear = np.random.uniform(-0.2, 0.2)
                
            else:
                # Default: mild deformation
                lambda_circ = np.random.uniform(0.9, 1.1)
                lambda_long = np.random.uniform(0.9, 1.1)
                lambda_rad = 1.0 / (lambda_circ * lambda_long)
                shear = 0.0
            
            # Construct strain tensor (diagonal + shear)
            if deformation_type == 'twist':
                C = np.array([
                    [lambda_circ**2, shear, 0],
                    [shear, lambda_long**2, 0],
                    [0, 0, lambda_rad**2]
                ])
            else:
                C = np.diag([lambda_circ**2, lambda_long**2, lambda_rad**2])
            
            # Add small noise
            C += np.random.normal(0, 0.01, (3, 3))
            C = (C + C.T) / 2  # Ensure symmetry
            
            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 0.1)  # Clamp eigenvalues
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            tensors.append(C)
        
        return np.array(tensors)  # (B, 3, 3)
    
    def compute_strain_features(
        self,
        positions: np.ndarray,
        fibers: np.ndarray,
        strain_tensor: np.ndarray
    ) -> np.ndarray:
        """
        Compute strain-derived features for each point.
        
        Args:
            positions: (Batch, N, 3)
            fibers: (Batch, N, 3)
            strain_tensor: (Batch, 3, 3)
        
        Returns:
            features: (Batch, N, 3) - [I1, I4, shear_strain]
        """
        batch_size, n_pts, _ = positions.shape
        features = np.zeros((batch_size, n_pts, 3))
        
        for b in range(batch_size):
            C = strain_tensor[b]
            
            # First invariant: I1 = tr(C)
            I1 = np.trace(C)
            
            for i in range(n_pts):
                f = fibers[b, i]
                
                # Fiber stretch: I4 = f^T C f
                I4 = f @ C @ f
                
                # Simple shear strain measure
                shear = np.abs(C[0, 1]) + np.abs(C[0, 2]) + np.abs(C[1, 2])
                
                features[b, i] = [I1, I4, shear]
        
        return features
    
    def generate(
        self,
        batch_size: int = 1,
        include_strain: bool = True,
        deformation_type: str = 'contraction'
    ) -> dict:
        """
        Generate complete synthetic cardiac dataset.
        
        Args:
            batch_size (int): Number of samples
            include_strain (bool): Whether to include strain features
            deformation_type (str): Type of cardiac deformation
        
        Returns:
            data: Dictionary containing:
                - 'positions': (B, N, 3) - 3D coordinates
                - 'fibers': (B, N, 3) - Fiber orientations
                - 'features': (B, N, 3) - Strain invariants
                - 'strain_tensors': (B, 3, 3) - Global strain tensors
                - 'observations': (B, N, 5 or 6) - Combined input/output
        """
        # Generate geometry
        positions = self.generate_cylindrical_points(batch_size)
        fibers = self.compute_fiber_orientations(positions)
        
        # Generate strain
        strain_tensors = self.generate_strain_tensor(batch_size, deformation_type)
        
        # Compute features
        if include_strain:
            features = self.compute_strain_features(positions, fibers, strain_tensors)
        else:
            features = np.zeros((batch_size, self.n_points, 3))
        
        # Create observation format
        # Input: (x, y, z, fiber_x, fiber_y, fiber_z) - actually using spherical coords for fibers
        # For simplicity, use first 2 components of fiber direction
        observations_input = np.concatenate([
            positions,  # (B, N, 3)
            fibers[:, :, :2]  # (B, N, 2) - simplified fiber representation
        ], axis=2)  # (B, N, 5)
        
        # Output: (x, y, z, I1, I4, shear)
        observations_output = np.concatenate([
            positions,
            features
        ], axis=2)  # (B, N, 6)
        
        return {
            'positions': positions,
            'fibers': fibers,
            'features': features,
            'strain_tensors': strain_tensors,
            'observations_input': observations_input,
            'observations_output': observations_output
        }
    
    def to_torch(self, data: dict) -> dict:
        """Convert numpy arrays to PyTorch tensors."""
        return {
            key: torch.from_numpy(val).float()
            for key, val in data.items()
        }


# Example usage
if __name__ == "__main__":
    print("Testing SynthCardioDataGenerator...")
    
    # Initialize generator
    generator = SynthCardioDataGenerator(
        n_points=100,
        noise_level=0.01,
        seed=42
    )
    
    # Generate data
    print("\nGenerating synthetic cardiac data...")
    data = generator.generate(batch_size=4, deformation_type='contraction')
    
    # Print shapes
    print("\nData shapes:")
    for key, val in data.items():
        print(f"  {key}: {val.shape}")
    
    # Verify properties
    print("\nVerifying properties...")
    
    # Check fiber normalization
    fibers = data['fibers']
    fiber_norms = np.linalg.norm(fibers, axis=2)
    print(f"Fiber norms range: [{fiber_norms.min():.4f}, {fiber_norms.max():.4f}]")
    assert np.allclose(fiber_norms, 1.0, atol=1e-2), "Fibers not normalized!"
    
    # Check strain tensor properties
    strain_tensors = data['strain_tensors']
    for i, C in enumerate(strain_tensors):
        # Check symmetry
        assert np.allclose(C, C.T, atol=1e-6), f"Strain tensor {i} not symmetric!"
        
        # Check positive definiteness
        eigvals = np.linalg.eigvalsh(C)
        assert np.all(eigvals > 0), f"Strain tensor {i} not positive definite!"
        
        # Check incompressibility (approximately)
        det_C = np.linalg.det(C)
        print(f"Strain tensor {i} determinant: {det_C:.4f} (should ≈ 1.0)")
    
    print("\n✓ All tests passed!")
    
    # Convert to PyTorch
    print("\nConverting to PyTorch tensors...")
    torch_data = generator.to_torch(data)
    print("✓ Conversion successful!")
    
    print("\nExample usage complete!")
