# Mathematical Background

## Differential Geometry

### Fiber Bundles
The cardiac geometry is modeled as a fiber bundle (E, π, M, F):
- **Base manifold M**: The 3D heart wall surface
- **Fiber F**: Space of fiber orientations at each point  
- **Total space E**: Combined space M × F
- **Projection π**: Maps total space to base

### Riemannian Metrics
A Riemannian metric G(z) on the latent space defines:
- Distance: d(z₁, z₂) = √((z₂-z₁)ᵀ G (z₂-z₁))
- Geodesics: Shortest paths satisfying the geodesic equation
- Christoffel symbols: Γⁱⱼₖ = ½ gⁱˡ(∂ⱼgₗₖ + ∂ₖgⱼₗ - ∂ₗgⱼₖ)

## Group Theory

### SO(3): Special Orthogonal Group
Represents all 3D rotations. Key properties:
- Closure: R₁R₂ ∈ SO(3)
- Identity: I ∈ SO(3)
- Inverses: R⁻¹ = Rᵀ
- Det(R) = 1

### Rodrigues' Formula
R = I + sin(θ)K + (1-cos(θ))K²

where K is the skew-symmetric matrix of the rotation axis.

## Topology

### Persistent Homology
Tracks topological features (connected components, holes, voids) across scales.

### Scene Graphs
Represent spatial relationships: (entities, relations)

## Strain Mechanics

### Cauchy-Green Tensor
C = FᵀF where F is the deformation gradient

### Invariants
- I₁ = tr(C): Volume change
- I₂ = ½(tr(C)² - tr(C²))
- I₃ = det(C): Incompressibility (I₃ ≈ 1)
- I₄ = fᵀCf: Fiber stretch
