# Tutorials

This folder contains the package's Jupyter tutorials. They are meant to be read roughly in the order below; each one builds on concepts introduced in the previous ones unless noted otherwise.

1. [GettingStarted.ipynb](GettingStarted.ipynb) — entry point: the `Potential` and `FDSolver` classes, basic band structures.
2. [AtomicToMolecular.ipynb](AtomicToMolecular.ipynb) — single-field `FDSolver`, the `dashboard` plotting function.
3. [EnergyLevels.ipynb](EnergyLevels.ipynb) — single-field `FDSolver`, the `energy_levels` plotting function.
4. [nDimensions.ipynb](nDimensions.ipynb) — working in 1D and 3D instead of 2D.
5. [PolaritonDispersion.ipynb](PolaritonDispersion.ipynb) — building a custom interactive plotting function from scratch.
6. [tightbinding.ipynb](tightbinding.ipynb) — cross-validation against the external `tightbinding` package (optional, requires a separate install).
7. [PillarsTETM.ipynb](PillarsTETM.ipynb) — multi-field/coupled equations, TE/TM polarization splitting.
8. [CouplingAPI.ipynb](CouplingAPI.ipynb) — the low-level coupling API used to build custom multi-field couplings like TE/TM.
9. [PlaneWaveSolver.ipynb](PlaneWaveSolver.ipynb) — the `PWSolver` class, a fast alternative to `FDSolver` for smooth potentials.
10. [Wannier.ipynb](Wannier.ipynb) — maximally localized Wannier functions, builds on `PWSolver`.

Most tutorials use 2D examples for simplicity, but the package supports 1D, 2D and 3D potentials and solvers throughout (see `nDimensions.ipynb`).
