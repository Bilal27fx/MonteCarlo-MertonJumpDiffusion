# Monte Carlo Merton Jump Diffusion Simulator

![Python](https://img.shields.io/badge/language-Python-blue) ![Jupyter](https://img.shields.io/badge/notebook-Jupyter-orange) ![License](https://img.shields.io/badge/license-MIT-green)

> A comprehensive Python toolkit for simulating the Merton Jump Diffusion model, performing option pricing, and visual analysis of asset paths with jumps.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Model Background](#model-background)
- [Project Structure](#project-structure)
- [Modules](#modules)
  - [Simulation Module](#simulation-module)
  - [Interactive App](#interactive-app)
  
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

---

## 🔎 Overview
The **Monte Carlo Merton Jump Diffusion Simulator** implements the classic Merton Jump Diffusion model, capturing both continuous Gaussian diffusion and discontinuous Poisson-driven jumps. It enables:

- Simulation of asset price paths under jumps and diffusion
- Estimation of European option prices and Greeks via Monte Carlo
- Visualization and sensitivity analysis of model parameters

Ideal for quantitative researchers, risk analysts, and anyone exploring jump–diffusion dynamics.

---

## 📈 Model Background
Merton’s Jump Diffusion augments the Black–Scholes framework by adding random jumps:

\[ dS_t = S_t\bigl((\mu - \lambda k)dt + \sigma dW_t\bigr) + S_t (J - 1)dN_t \]

- **$\mu$**: drift rate  
- **$\sigma$**: diffusion volatility  
- **$\lambda$**: jump intensity (expected jumps per year)  
- **$J$**: jump multiplier, with $\ln J \sim \mathcal{N}(\mu_J, \sigma_J^2)$  
- **$k = E[J - 1]$**: average jump size minus one  
- **$N_t$**: Poisson process with rate $\lambda$  

This model captures fat tails and skewness in returns.

---

## 🗂 Project Structure
```
MonteCarlo-MertonJumpDiffusion/
├── Simulation.py        # Core MJD simulation engine
├── app.py               # Interactive visualization (Streamlit or Flask)
├── fichier.ipynb        # Jupyter notebook with examples
├── Pictures/            # Sample plots and output figures
├── requirements.txt     # Python package dependencies
└── README.md            # Project overview and usage
```

---

## ⚙️ Modules

### Simulation Module
**File**: `Simulation.py`  
Implements:
- Parameter setup: drift $\mu$, volatility $\sigma$, jump intensity $\lambda$, jump distribution $(\mu_J, \sigma_J)$.  
- Monte Carlo engine: generates asset paths, applies payoff functions.  
- Output: CSV/JSON of simulated paths, option price estimates, standard errors.

### Interactive App
**File**: `app.py`  
Provides a web-based UI for:
- Inputting model parameters on the fly.  
- Launching simulations and viewing real-time results.  
- Displaying path trajectories, payoff distributions, and sensitivity charts.

---

## 🛠 Installation
1. Clone the repo:  
   ```bash
   git clone https://github.com/Bilal27fx/MonteCarlo-MertonJumpDiffusion.git
   ```  
2. Navigate and install dependencies:  
   ```bash
   cd MonteCarlo-MertonJumpDiffusion
   pip install -r requirements.txt
   ```  
3. (Optional) Create and activate a virtual environment for isolation.

---

## 🚀 Usage

To launch the interactive dashboard, simply run:

```bash
streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.
---

## 🛠 Configuration
- **`config.yaml`** (if used): specify default parameters, number of paths, time grid.  
- **Environment variables** (if needed) for advanced settings.

---

## 📊 Examples
See the **Pictures/** folder for:
- Sample asset paths with varying jump intensities  
- Histograms of terminal price distributions  
- Option price surfaces vs. strike and maturity  

---

## 🤝 Contributing
Contributions are welcome!  
1. Fork the project.  
2. Create a feature branch (`git checkout -b feature/xyz`).  
3. Commit your changes (`git commit -m 'Add new functionality'`).  
4. Push to branch (`git push origin feature/xyz`).  
5. Open a Pull Request.  

Please follow the [code of conduct](CODE_OF_CONDUCT.md) and add tests for new features.

---

## ⚖️ License
This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

