README.md
# Quantum Error Correction Surface Code Simulator 🧬

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)
[![Stim](https://img.shields.io/badge/Stim-1.0+-green)](https://github.com/quantumlib/Stim)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/samkelomhlophe/qec-surface-code)](https://github.com/samkelomhlophe/qec-surface-code)

**Python implementation of a rotated surface code** with:
- Stim circuit generation
- PyMatching MWPM decoder
- Neural-network decoder (Torch)
- Realistic depolarizing noise
- Threshold analysis & comparison plots

## Quick Start
```bash
pip install stim pymatching torch numpy matplotlib
python main.py
