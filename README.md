# DeSIDE-DDI 🚀

Welcome to the modernized version of **DeSIDE-DDI**, a deep-learning strategy for the interpretable prediction of Drug-Drug Interactions (DDIs) leveraging drug-induced gene expression signatures.

The original methodology was developed by Eunyoung Kim. The codebase has been fully upgraded to **TensorFlow 2.x** and enhanced with a beautiful, interactive **Web UI Interface** for streamlined side-effect discovery.

## ✨ Key Upgrades 
- **TensorFlow 2.x Architecture:** Seamless execution on macOS Apple Silicon (M-series Metal), Windows CUDA, and Linux natively. No obsolete `tf.contrib` or TF1 sessions.
- **Interactive Web UI:** Added a stunning, modern light-themed Glassmorphism web application powered by a blazing fast Flask backend that pre-loads models.
- **One-Click Launch:** Scripts added for instantaneous execution across any operating system without worrying about virtual environments.
- **IDE-Friendly Python Scripts:** Legacy Jupyter Notebooks converted directly into robust CLI scripts (`feature_generation.py`, `ddi_prediction.py`, `feature_analysis.py`).

---

## 💻 Quick Start (Web Application)

You do not need to manually create virtual environments or install pip packages. The included scripts will automatically configure everything and open the AI Engine in your browser!

### For macOS and Linux
```bash
chmod +x run_mac_linux.sh
./run_mac_linux.sh
```

### For Windows
Double-click `run_windows.bat` or run it from your command prompt:
```cmd
run_windows.bat
```

> **Note:** The UI runs on Port 5002 dynamically pre-loading the 963 possible side effects to provide instant predictions upon submitting a drug pair.

---

## 🛠️ Background Execution (CLI)

If you prefer to interface with the original pipeline or run tests directly via the command line:

```bash
# Activate the environment initialized by the launch scripts
source venv/bin/activate  # On Windows: venv\Scripts\activate.bat

# Feature Generation Model
# Extracts significant genes given drug pairs (Compound fingerprints & properties)
python feature_generation.py

# DDI Prediction Model
# Evaluates predicted gene expressions and yields side effect scores
python ddi_prediction.py

# Feature Analysis
# Visualizes changed latent features locally to images
python feature_analysis.py
```

## Original Colab Resources
- **Feature generation model:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1m7fyZwFPp_85wKvFjbCwRCIOScrJEUtX?usp=sharing)
- **DDI prediction model:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XslE3XNsjm-dXwxrk_eVST6kALdwmvkd?usp=sharing)

*Originally developed by Eunyoung Kim.*
