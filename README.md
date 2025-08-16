# Cost-Functions-And-Activations-NeuralNetworks

# Initialization & Loss Function Study (PyTorch + Iris)
**Notebook:** `NeuralNetworksCostFunctionsAndActivations.ipynb`  
**Subtitle:** ~ By Guardians of The Galaxy

This project explores how **weight initialization** (Constant, Uniform, Xavier) and **loss functions** (**CrossEntropy**, **NLL**) affect optimization dynamics and final performance on a classic tabular dataset (**Iris**, via `sklearn.datasets.load_iris`). Implemented in **PyTorch** with a compact linear classifier and **Adam** optimizer. The notebook also visualizes training/validation curves.

---

## 🎯 Objectives
- Compare **initialization schemes**: Constant, Uniform, Xavier.
- Compare **losses**: `CrossEntropyLoss` vs `NLLLoss` (with `LogSoftmax`).
- Track **training/validation accuracy**, **time-to-convergence**, and **epochs**.
- Plot **loss** and **accuracy** curves per experiment.

---

## 🧪 Data
- **Dataset:** `sklearn.datasets.load_iris()` (3 classes, 4 numeric features).
- **Split:** training/validation split inside the notebook.
- **Preprocessing:** standard tabular normalization if enabled in the code (see data prep cell).

---

## 🧠 Model & Training
- **Framework:** PyTorch (CPU)
- **Architecture:** compact linear classifier (`nn.Linear` layers), no convolution.
- **Optimizer:** `Adam`
- **Loss:** `CrossEntropyLoss` or `NLLLoss` (with `LogSoftmax` in the model head)
- **Epochs:** experiments converged within ~30–100 epochs depending on init/loss.
- **Device:** CPU (no CUDA used in the current notebook run).

---

## 📈 Results (from notebook run)
| Experiment | Train Acc | Val Acc | Time (s) | Epoch |
|---|---:|---:|---:|---:|
| Constant Initialization + CrossEntropy Loss | 1.0000 | 1.0000 | 0.67 | 99 |
| Constant Initialization + NLL Loss | 1.0000 | 1.0000 | 0.22 | 99 |
| Uniform Initialization + CrossEntropy Loss | 1.0000 | 1.0000 | 0.14 | 58 |
| Uniform Initialization + NLL Loss | 1.0000 | 1.0000 | 0.11 | 31 |
| Xavier Initialization + CrossEntropy Loss | 1.0000 | 1.0000 | 0.20 | 57 |
| Xavier Initialization + NLL Loss | 1.0000 | 1.0000 | 0.20 | 58 |


> On Iris, all settings achieve **perfect accuracy**; differences are mainly in **training time** and **convergence speed**, with **Uniform/Xavier** generally converging faster than **Constant**.

---

## 📊 Visualizations
- Training **loss** vs. epoch
- Validation **accuracy** vs. epoch

These plots are generated inline. To include them in the repo, save figures to an `images/` directory and link them in this README.

---

## 📁 Repository Structure
```text
├── NeuralNetworksCostFunctionsAndActivations.ipynb   # Main analysis notebook
└── README.md         # This file
```

---

## ⚙️ Environment & Setup
**Core libraries:** `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`

Install:
```bash
pip install -U torch torchvision torchaudio numpy pandas scikit-learn matplotlib
```

Run:
```bash
jupyter notebook "NeuralNetworksCostFunctionsAndActivations.ipynb"
```

---

## 💡 Takeaways & Next Steps
- On linearly separable or easy datasets like **Iris**, multiple setups reach **100%**; focus on **stability** and **speed** rather than raw accuracy.
- Try **Kaiming (He) initialization** and compare with Xavier/Uniform.
- Add **regularization** (Dropout, Weight Decay) to test robustness.
- Evaluate on **harder datasets** (e.g., Wine, Breast Cancer) or noisy synthetic data.
- Log detailed metrics (e.g., precision/recall/F1) and a **confusion matrix** for per-class insight.
