"""Shared fixtures. Keeps the suite headless and quiet."""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _quiet_display(monkeypatch):
    """The plotting functions display ipywidgets on call, which is noise outside a kernel."""
    import bloch_schrodinger.plotting as plotting

    monkeypatch.setattr(plotting, "display", lambda *a, **k: None)


def harmonic(potential, omegas):
    """Set an anisotropic harmonic trap, whose spectrum is known analytically."""
    potential.set(
        sum(
            omegas[i] ** 2 / 2 * coo**2 for i, coo in enumerate(potential.coords)
        )
    )
    return potential


def analytic_levels(omegas, n_levels):
    """The lowest n_levels of sum_i omega_i (n_i + 1/2), for alpha = hbar^2/2m = 1/2."""
    grid = np.arange(12)
    mesh = np.meshgrid(*[grid] * len(omegas), indexing="ij")
    energies = sum(
        omegas[i] * (mesh[i] + 0.5) for i in range(len(omegas))
    ).ravel()
    return np.sort(energies)[:n_levels]
