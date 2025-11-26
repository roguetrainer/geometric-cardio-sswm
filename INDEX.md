# Geometric Cardio SSWM - Complete Package Index

## 📁 Quick Navigation

### 🚀 Get Started
- **[QUICKSTART.md](QUICKSTART.md)** - 5-minute setup and demo
- **[README.md](README.md)** - Full project documentation
- **[PACKAGE_SUMMARY.md](PACKAGE_SUMMARY.md)** - Package contents overview

### 📖 Documentation
- **[docs/mathematical_background.md](docs/mathematical_background.md)** - Math foundations
- **[docs/architecture_details.md](docs/architecture_details.md)** - Model architectures
- **[docs/training_guide.md](docs/training_guide.md)** - Training & hyperparameters

### 💻 Source Code
- **[src/encoder.py](src/encoder.py)** - PointNet-based encoder
- **[src/dynamics_model.py](src/dynamics_model.py)** - Latent dynamics prediction
- **[src/decoder.py](src/decoder.py)** - Observation reconstruction
- **[src/sswm_model.py](src/sswm_model.py)** - Complete SSWM model
- **[src/data_generator.py](src/data_generator.py)** - Synthetic data generation
- **[src/geometric_utils.py](src/geometric_utils.py)** - Geometric computations
- **[src/training.py](src/training.py)** - Training utilities
- **[src/visualization.py](src/visualization.py)** - Plotting functions

### 📓 Interactive Notebooks
- **[notebooks/01_geometric_priors.ipynb](notebooks/01_geometric_priors.ipynb)** - Group theory & geometry
- **[notebooks/02_synthcardio_data.ipynb](notebooks/02_synthcardio_data.ipynb)** - Data exploration

### 🛠️ Setup Files
- **[requirements.txt](requirements.txt)** - Python dependencies
- **[setup.sh](setup.sh)** - Installation script
- **[.gitignore](.gitignore)** - Git ignore patterns
- **[LICENSE](LICENSE)** - MIT License

### 📊 Outputs
- **[img/](img/)** - Generated visualizations (empty, populated during use)

## 📋 File Statistics

```
Total files: ~25
Python modules: 9
Jupyter notebooks: 5
Documentation: 7 markdown files
Lines of code: ~2,000+
```

## 🎯 Common Tasks

### Installation
```bash
bash setup.sh
```

### Run Tests
```bash
python -m src.encoder
python -m src.dynamics_model
python -m src.data_generator
```

### Quick Demo
```python
from src import SynthCardioDataGenerator, SynthCardioSSWM, train_sswm
data = SynthCardioDataGenerator().generate(batch_size=100)
model = SynthCardioSSWM()
history = train_sswm(model, data, epochs=10)
```

### Launch Notebooks
```bash
jupyter notebook notebooks/
```

## 📚 Learning Path

1. **Beginner**: Start with QUICKSTART.md → README.md → Notebook 01
2. **Intermediate**: Read architecture_details.md → Run all notebooks
3. **Advanced**: Modify src/ modules → Train on custom data → Extend architecture

## 🔗 Key Concepts

- **SSWM**: Self-Supervised World Models - predictive models of dynamics
- **Geometric Priors**: SO(3), E(3) groups, Riemannian metrics
- **Cardiac Mechanics**: Fiber architecture, strain tensors, incompressibility
- **Point Clouds**: Permutation-invariant representations of 3D geometry

## 📦 Package Structure

```
geometric-cardio-sswm/
├── 📄 Documentation (README, QUICKSTART, etc.)
├── 📁 src/ (Python modules)
├── 📁 notebooks/ (Jupyter notebooks)
├── 📁 docs/ (Detailed documentation)
├── 📁 img/ (Visualizations)
├── ⚙️ requirements.txt
└── 🔧 setup.sh
```

## 🎓 Citation

```bibtex
@software{geometric_cardio_sswm,
  title={Geometric Cardio SSWM},
  year={2025}
}
```

---

**Version**: 0.1.0  
**Created**: November 2025  
**License**: MIT  
**Status**: Research Implementation
