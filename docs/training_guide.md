# Training Guide

## Quick Start

```python
from src.sswm_model import SynthCardioSSWM
from src.data_generator import SynthCardioDataGenerator
from src.training import train_sswm

# Generate data
gen = SynthCardioDataGenerator(n_points=100)
data = gen.generate(batch_size=1000)

# Create model
model = SynthCardioSSWM(n_points=100, latent_dim=32)

# Train
history = train_sswm(model, data, epochs=100)
```

## Hyperparameter Tuning

### Latent Dimension
- **16-32**: For simple geometries
- **32-64**: For complex cardiac mechanics
- **64-128**: For high-resolution data

### Learning Rate
- Start with 1e-3
- Reduce by 10× if loss plateaus
- Use learning rate scheduling

### Beta (KL Weight)
- **0.0001-0.001**: Standard VAE regularization
- Higher β → more regularized latent space
- Lower β → better reconstruction

## Monitoring Training

Watch for:
- Reconstruction loss decreasing steadily
- KL loss stabilizing (not collapsing to 0)
- Predictions maintaining physical plausibility

## Common Issues

### Mode Collapse
Symptoms: KL loss → 0, poor diversity
Solution: Increase β, add capacity

### Poor Reconstruction  
Symptoms: High MSE, blurry predictions
Solution: Decrease β, increase model capacity

### Unstable Training
Symptoms: Loss oscillations, NaN values
Solution: Reduce learning rate, gradient clipping
