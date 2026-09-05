# Bloch-Schrödinger solver 

This package provides the tools to solve the Bloch-Schrödinger equation in 1D, 2D and 3D. It contains potential construction tools, a finite-difference solver (`FDSolver`, including support for coupled multi-field equations such as TE/TM polarization splitting), a plane-wave solver (`PWSolver`) for smooth potentials, and a `Wannier` class for computing maximally localized Wannier functions.

## Installation

First download the repository and extract it where you want. Then, run in your python environment 

`bash`
pip install path\\to\\package\\bloch_schrodinger

or 

`bash`
pip install -e path\\to\\package\\bloch_schrodinger

if you want to be able to modify it in place.

## Getting started

Once you have installed the package, open in a jupyter viewer the [Getting Started](docs/GettingStarted.ipynb) notebook. For the full list of tutorials and a suggested reading order, see [docs/README.md](docs/README.md).



