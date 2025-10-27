# Graphene FET Virtual Laboratory

A virtual laboratory for simulating and analyzing Graphene Field-Effect Transistors (GFETs) using Verilog-A models and Python analysis tools.

## Overview

This project provides tools for simulating graphene FET devices, including:
- Verilog-A compact model for circuit simulation
- Python interface for SPICE simulation (PySpice/Ngspice)
- Jupyter notebooks for I-V characteristic analysis
- Parameter sweep capabilities

## Project Structure

```
graphene-fet-lab/
├── models/
│   └── graphene_fet.sv         # Verilog-A GFET model
├── notebooks/
│   └── GFET_IV_curves.ipynb   # Jupyter notebook for analysis
├── src/
│   └── spice_interface.py      # Python SPICE interface
├── README.md
└── requirements.txt
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Install PySpice for SPICE simulation
pip install PySpice
```

## Usage

### Jupyter Notebook Analysis

```bash
jupyter notebook notebooks/GFET_IV_curves.ipynb
```

The notebook includes:
- Analytical GFET model
- Output characteristics (Ids vs Vds)
- Transfer characteristics (Ids vs Vgs)
- Ambipolar conduction analysis
- Parameter sensitivity studies

### Python Interface

```bash
python src/spice_interface.py
```

This runs a demo showing transfer and output curves.

### Verilog-A Model

The `models/graphene_fet.sv` file can be compiled and used with:
- Ngspice (with ADMS)
- Cadence Spectre
- Synopsis HSPICE (with Verilog-A support)

## Device Physics

### Key Features

- **Ambipolar Conduction**: Both electron and hole conduction
- **Dirac Point**: Minimum conductivity at Vgs ≈ 0V
- **High Mobility**: Typical values 5,000-40,000 cm²/V·s
- **Linear Band Structure**: Results in high-frequency performance

### Model Parameters

- `W`: Channel width (m)
- `L`: Channel length (m)
- `Cox`: Gate oxide capacitance (F/cm²)
- `mu`: Carrier mobility (cm²/V·s)
- `Vdirac`: Dirac point voltage (V)
- `vF`: Fermi velocity (m/s)

## Applications

- RF amplifiers
- High-speed digital circuits
- Sensors and biosensors
- Flexible electronics
- Optoelectronics

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- Jupyter
- PySpice (optional, for SPICE integration)

## References

1. Jiménez, D. (2012). "Drift-diffusion model for graphene field-effect transistors." IEEE TED
2. Meric, I. et al. (2008). "Current saturation in zero-bandgap, top-gated graphene field-effect transistors." Nature Nanotechnology

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or pull request.
