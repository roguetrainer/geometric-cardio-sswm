# Quick Start Guide

## Installation

```bash
# Navigate to the project directory
cd geometric-cardio-sswm

# Run setup script
bash setup.sh

# Or install manually
pip install -r requirements.txt
```

## 5-Minute Demo

```python
# 1. Import modules
from src.data_generator import SynthCardioDataGenerator
from src.sswm_model import SynthCardioSSWM
from src.training import train_sswm
from src.visualization import plot_point_cloud

# 2. Generate synthetic data
generator = SynthCardioDataGenerator(n_points=100, seed=42)
data = generator.generate(batch_size=100)

# 3. Create model
model = SynthCardioSSWM(
    n_points=100,
    latent_dim=32,
    hidden_dim=64
)

# 4. Train (quick test)
history = train_sswm(
    model, 
    data, 
    epochs=10,
    batch_size=16,
    verbose=True
)

# 5. Visualize
import matplotlib.pyplot as plt
positions = data['positions'][0]
fibers = data['fibers'][0]
plot_point_cloud(positions, fibers, title="Cardiac Geometry")
plt.show()

print("Success! Model trained and data visualized.")
```

## Explore Notebooks

```bash
jupyter notebook notebooks/
```

Start with:
1. `01_geometric_priors.ipynb` - Mathematical foundations
2. `02_synthcardio_data.ipynb` - Data exploration

## Test Installation

```bash
# Run module tests
python -m src.encoder
python -m src.dynamics_model
python -m src.decoder
python -m src.sswm_model
python -m src.data_generator
python -m src.geometric_utils
```

All tests should pass with "✓ All tests passed!" messages.

## Project Structure

```
geometric-cardio-sswm/
├── src/              # Python modules
├── notebooks/        # Jupyter notebooks  
├── docs/            # Documentation
├── img/             # Generated images
├── README.md        # Main documentation
└── requirements.txt # Dependencies
```

## Key Components

- **Encoder**: Maps point clouds → latent vectors
- **Dynamics Model**: Predicts latent evolution
- **Decoder**: Reconstructs observations
- **Data Generator**: Creates synthetic cardiac data
- **Geometric Utils**: Mathematical operations

## Next Steps

1. Read `README.md` for comprehensive overview
2. Explore `docs/` for detailed documentation
3. Run notebooks to understand concepts
4. Modify hyperparameters and experiment
5. Extend to your own data

## Getting Help

- Check documentation in `docs/`
- Review code comments in `src/`
- Run individual module tests
- Open an issue on GitHub

## Citation

If using this code, please cite:

```bibtex
@software{geometric_cardio_sswm,
  title={Geometric Cardio SSWM},
  year={2025},
  author={},
  url={https://github.com/yourusername/geometric-cardio-sswm}
}
```

---

Happy modeling! 🫀🧠
