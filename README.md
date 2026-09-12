# Quantum Error Correction Surface Code Simulator

A Python project for experimenting with **surface-code quantum error correction**, noise models, and classical decoding approaches.

The project explores how logical quantum information can be protected from physical errors by simulating noisy surface-code circuits and comparing different decoding strategies.

## Why I Built This

I am a BICT (Internet of Things) student with a long-term interest in quantum computing and AI. I built this project to move beyond circuit-level experimentation and learn more about quantum error correction, noise, decoding, and the engineering problems involved in building fault-tolerant quantum systems.

## Features

- Surface-code circuit generation with **Stim**
- Minimum-weight perfect matching decoding with **PyMatching**
- Experimental neural-network decoder built with **PyTorch**
- Depolarizing-noise modelling
- Logical-error-rate experiments
- Threshold-analysis utilities
- Decoder-comparison and result-visualisation scripts

## Technology

- Python
- Stim
- PyMatching
- PyTorch
- NumPy
- Matplotlib
- tqdm

## Project Structure

```text
qec-surface-code/
├── src/
│   ├── surface_code.py          # Surface-code circuit construction
│   ├── decoder.py               # Matching-based decoding
│   ├── nn_decoder.py            # Experimental neural-network decoder
│   ├── noise_model.py           # Noise-model utilities
│   └── threshold_analysis.py    # Threshold experiments
├── data/                        # Experiment data
├── figures/                     # Generated plots and figures
├── paper/                       # Research notes / project material
├── plot_decoder_comparison.py   # Decoder comparison visualisation
├── plot_results.py              # Result plotting
├── requirements.txt
└── README.md
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/samkelomhlophe/qec-surface-code.git
cd qec-surface-code
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Project

The repository contains separate scripts for simulation, decoding, threshold analysis and plotting. Start by inspecting the modules in `src/` and the plotting scripts in the project root.

> Note: `main.py` is currently a placeholder while the project is being reorganised into a cleaner experiment workflow.

## What I Learned

This project has helped me develop practical familiarity with:

- quantum error-correction concepts
- surface codes and logical errors
- simulation-based experimentation
- classical decoding of quantum syndromes
- comparing algorithmic and neural approaches
- scientific Python tooling
- organising reproducible technical experiments

## Current Status

This is an **active learning and research project**, not a production quantum-error-correction implementation. I am continuing to improve the experiment pipeline, documentation and comparison methodology.

## Author

**Samkelo Mhlophe**  
BICT (Internet of Things), Durban University of Technology  
Interested in AI, software engineering, IoT and quantum computing

GitHub: https://github.com/samkelomhlophe

## License

MIT License
