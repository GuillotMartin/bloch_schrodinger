"""Tests of the Wannier -> tight-binding pipeline.

Like the rest of the suite, these are anchored to conditions that must hold rather than to current output:
with no coupling dropped, the tight-binding bands reproduce the Bloch energies exactly on the k-mesh; the
amplitudes are hermitian; and the labels of the couplings agree with an independent real-space computation.
"""

import numpy as np
import pytest

pytest.importorskip("tightbinding")

from tightbinding.geometry import Lattice, Orbital, Site, Unitcell  # noqa: E402

from bloch_schrodinger.potential import Potential, create_parameter  # noqa: E402
from bloch_schrodinger.wannier_tb import WannierHamiltonianBuilder  # noqa: E402

L = 4.0
A_HONEY = 4 * np.pi / 3**1.5 / 2


def cosine_1d(resolution=(48,)):
    p = Potential(unitvecs=[[L]], resolution=resolution, v0=0)
    p.set(-20 * np.cos(2 * np.pi * p.coords[0] / L))
    return p


def honeycomb(resolution=(32, 32), depth=6, delta=0):
    kl, a = 2, A_HONEY
    a1 = np.array([3 * a / 2, -(3**0.5) * a / 2])
    a2 = np.array([3 * a / 2, 3**0.5 * a / 2])
    ks = [
        kl * np.array([-(3**0.5) / 2, 1 / 2]),
        kl * np.array([3**0.5 / 2, 1 / 2]),
        kl * np.array([0, -1]),
    ]
    p = Potential(unitvecs=[a1, a2], resolution=resolution, v0=0)
    dirs = [k[0] * (p.x - a1[0]) + k[1] * p.y for k in ks]
    for i in range(3):
        # The two triangular sublattices, 'delta' deepening one and raising the other
        p.add(-(depth + delta) * np.cos((dirs[i - 1] - dirs[i]) - 2 * np.pi / 3))
        p.add(-(depth - delta) * np.cos((dirs[i - 1] - dirs[i]) + 2 * np.pi / 3))
    return p


def honeycomb_lattice(pot, order=("A", "B"), cells=((0, 0),)):
    s = Orbital("s")
    position = {"A": [-A_HONEY / 2, 0], "B": [A_HONEY / 2, 0]}
    uc = Unitcell("honeycomb", pot.a)
    uc.add_site([Site(name, position[name], s) for name in order])
    lattice = Lattice("honeycomb", uc, (True, True))
    for cell in cells:
        lattice.add_unitcell(cell)
    lattice.define_index()
    return lattice


def build_honeycomb(pot=None, lattice=None, **kwargs):
    pot = honeycomb() if pot is None else pot
    lattice = honeycomb_lattice(pot) if lattice is None else lattice
    options = dict(k_resolution=7, threshold=0, seed=0, verbose=False, blockwargs={"E_lim": 300})
    options.update(kwargs)
    return WannierHamiltonianBuilder(pot, 0.5, lattice, **options)


def mp_bands(H):
    """The tight-binding bands on the Monkhorst-Pack mesh of the Wannier functions, and the Bloch energies there."""
    H = H.on_lattice()  # a copy, so that the fixtures are not modified
    kb_dims = H.wannier.kb_dims
    k = [kc.transpose(*kb_dims).values.ravel() for kc in H.wannier.k]
    H.add_parameters({"kpt": np.arange(k[0].size)})
    for name, kc in zip(H.recoords, k):
        H.add_multidimensional_coord(name, kc, ("kpt",))
    H.build()
    eigva, _ = H.eigh()
    tb = eigva.transpose(*H.param_dims, "kpt", "band").values
    ref = H.wannier.eigva.transpose(*H.param_dims, *kb_dims, "m").values.reshape(tb.shape)
    return tb, ref


@pytest.fixture(scope="module")
def honey():
    return build_honeycomb()


# --------------------------------------------------------------------------------------
# Exactness on the k-mesh
# --------------------------------------------------------------------------------------


def test_1d_bands_are_exact_on_the_k_mesh():
    pot = cosine_1d()
    uc = Unitcell("chain", [[L]])
    uc.add_site(Site("A", [0.0], Orbital("s")))
    lattice = Lattice("chain", uc, (True,))
    lattice.add_unitcell((0,), update=True)

    H = WannierHamiltonianBuilder(
        pot, 0.5, lattice, k_resolution=9, threshold=0, seed=0, verbose=False,
        blockwargs={"E_lim": 150},
    )
    tb, ref = mp_bands(H)
    assert np.abs(tb - ref).max() < 1e-8

    # With a single band, the on-site energy is the band average
    assert float(H.tb_parameters["E_A_s"]) == pytest.approx(float(H.wannier.eigva.mean()), rel=1e-10)

    t = H.hoppings.sel(orb1="A_s", orb2="A_s")
    for r in range(1, 5):
        assert complex(t.sel(R1=r)) == pytest.approx(np.conj(complex(t.sel(R1=-r))), abs=1e-12)


def test_honeycomb_bands_are_exact_on_the_k_mesh(honey):
    tb, ref = mp_bands(honey)
    assert np.abs(tb - ref).max() < 1e-8


def test_parameter_dimensions_are_carried_to_the_hamiltonian():
    depth = create_parameter("depth", [5, 7])
    H = build_honeycomb(pot=honeycomb(depth=depth))
    assert H.param_dims == ["depth"]
    assert list(H.tb_parameters["t_A_s_B_s_0_0"].dims) == ["depth"]

    tb, ref = mp_bands(H)
    assert np.abs(tb - ref).max() < 1e-8
    # A deeper lattice tunnels less
    t_nn = abs(H.tb_parameters["t_A_s_B_s_0_0"])
    assert float(t_nn.sel(depth=7)) < float(t_nn.sel(depth=5))


# --------------------------------------------------------------------------------------
# Labels: centers, sites and displacements
# --------------------------------------------------------------------------------------


def test_centers_agree_with_the_real_space_profiles(honey):
    tiled, wf = honey.wannier.compute_wannier(honey.U_mnk, [(-2, 3), (-2, 3)], verbose=False)
    density = abs(wf) ** 2 * tiled.get_dS()
    for c, name in enumerate(["x", "y"]):
        mean = (density * wf.coords[name]).sum(["a1", "a2"]).values
        assert np.allclose(mean, honey.centers.isel(axis=c).values, atol=1e-3)
    assert np.allclose(honey.centers.values, honey.positions, atol=1e-3)


def test_site_order_does_not_matter():
    # An imbalanced lattice, so that the two on-site energies cannot be mistaken for one another
    pot = honeycomb(delta=1)
    reference = build_honeycomb(pot=pot)
    H = build_honeycomb(pot=pot, lattice=honeycomb_lattice(pot, order=("B", "A")))
    assert H.orbitals == ["B_s", "A_s"]
    gap = abs(float(reference.tb_parameters["E_A_s"] - reference.tb_parameters["E_B_s"]))
    assert gap > 0.1

    tb, ref = mp_bands(H)
    assert np.abs(tb - ref).max() < 1e-8
    for name in ["E_A_s", "E_B_s"]:
        assert float(H.tb_parameters[name]) == pytest.approx(float(reference.tb_parameters[name]), abs=1e-3 * gap)


def test_a_wannier_function_left_in_another_cell_is_brought_back(honey):
    H = honey.on_lattice()
    a1 = H.wannier.potential.a[0]
    kx, ky = [kc.transpose(*H.wannier.kb_dims) for kc in H.wannier.k]

    # Multiplying u_nk by exp(ik.a1) translates the second WF by +a1
    shifted = H.U_mnk.copy()
    shifted[{"n": 1}] = shifted[{"n": 1}] * np.exp(1j * (kx * a1[0] + ky * a1[1]))
    assert np.allclose(H.wannier.centers(shifted, verbose=False).isel(n=1), H.positions[1] + a1, atol=1e-3)

    realigned = H._align(shifted, verbose=False)
    assert np.allclose(realigned.values, H.U_mnk.values, atol=1e-10)


def test_couplings_match_real_space_integrals(honey):
    # Within this cutoff, A->B at (-1, 0) is a nearest neighbor and A->B at (1, 0) a third neighbor:
    # a displacement convention inverted somewhere would swap them
    H = honey.on_lattice(cutoff=3.3)
    assert abs(float(H.tb_parameters["t_A_s_B_s_m1_0"])) > 10 * abs(float(H.tb_parameters["t_A_s_B_s_1_0"]))

    real_space = H.real_space_hoppings(verbose=False)
    for name in H.tb_parameters.data_vars:
        # The finite-difference Hamiltonian carries a discretization error of a fraction of a percent
        assert float(real_space[name]) == pytest.approx(float(H.tb_parameters[name]), rel=0.03, abs=1e-3)


# --------------------------------------------------------------------------------------
# Selection of the couplings
# --------------------------------------------------------------------------------------


def test_the_shells_are_those_of_the_honeycomb(honey):
    expected = A_HONEY * np.array([1, 3**0.5, 2, 7**0.5, 3])
    assert np.allclose(honey.shell_distances[:5], expected)


def test_neighbors_selects_whole_shells(honey):
    first = honey.on_lattice(neighbors=1)
    assert set(first.tb_parameters.data_vars) == {
        "E_A_s", "E_B_s", "t_A_s_B_s_0_0", "t_A_s_B_s_m1_0", "t_A_s_B_s_0_m1"
    }
    assert first.tb_parameters["t_A_s_B_s_0_0"].attrs["shell"] == 1

    second = honey.on_lattice(neighbors=2)
    added = set(second.tb_parameters.data_vars) - set(first.tb_parameters.data_vars)
    assert len(added) == 6
    assert all(name.startswith(("t_A_s_A_s", "t_B_s_B_s")) for name in added)


def test_neighbors_keeps_every_bond_of_the_shells_however_weak(honey):
    n = len(honey.shell_distances)
    H = honey.on_lattice(neighbors=n, threshold=None)

    # Every bond within the n-th shell, counted from the distances: each A-B bond once, and each A-A or
    # B-B bond once out of the +R/-R pair
    limit = honey.shell_distances[-1] * (1 + 1e-9)
    d = honey.distances
    n_AB = int((d.sel(orb1="A_s", orb2="B_s") <= limit).sum())
    n_same = sum(
        int(((d.sel(orb1=o, orb2=o) <= limit) & (d.sel(orb1=o, orb2=o) > 0)).sum()) // 2
        for o in ["A_s", "B_s"]
    )
    assert len(H.coupling_table) == 2 + n_AB + n_same

    # ...including bonds that the default magnitude threshold, 1e-3, would have dropped
    hoppings = [c["amplitude"] for c in H.coupling_table if c["shell"] != 0]
    assert min(hoppings) < 1e-3 * max(hoppings)
    assert len(honey.on_lattice(neighbors=n, threshold=1e-3).coupling_table) < len(H.coupling_table)


def test_too_many_shells_for_the_k_mesh_are_rejected(honey):
    with pytest.raises(ValueError, match="k_resolution"):
        honey.on_lattice(neighbors=len(honey.shell_distances) + 1)


def test_a_new_distance_criterion_replaces_the_previous_one(honey):
    H = honey.on_lattice(cutoff=1.5)
    assert honey.on_lattice().cutoff is None
    assert H.on_lattice().cutoff == 1.5
    switched = H.on_lattice(neighbors=2)
    assert switched.cutoff is None
    assert len(switched.coupling_table) == 11


# --------------------------------------------------------------------------------------
# Reuse of the parameters on other lattices
# --------------------------------------------------------------------------------------


def test_a_supercell_folds_the_primitive_bands(honey):
    pot = honey.wannier.potential
    supercell = honeycomb_lattice(pot, cells=((0, 0), (1, 0)))
    k = np.array([0.31, -0.17])
    b1 = 2 * np.pi * np.linalg.inv(pot.a).T[0]

    # The primitive bands at k and k + b1/2, both points being coordinates of a common dimension 'q'
    primitive = honey.on_lattice(params={"q": np.arange(2)}, threshold=1e-4)
    primitive.add_multidimensional_coord("kx", np.array([k[0], k[0] + b1[0] / 2]), ("q",))
    primitive.add_multidimensional_coord("ky", np.array([k[1], k[1] + b1[1] / 2]), ("q",))
    primitive.build()
    folded = np.sort(primitive.eigh()[0].values.ravel())

    H = honey.on_lattice(lattice=supercell, params={"kx": k[0], "ky": k[1]}, threshold=1e-4)
    H.build()
    assert H.Hamiltonian.sizes["i"] == 4
    assert np.allclose(np.sort(H.eigh()[0].values.ravel()), folded, atol=1e-10)


# --------------------------------------------------------------------------------------
# Input validation
# --------------------------------------------------------------------------------------


def test_mismatched_unit_vectors_are_rejected():
    pot = honeycomb()
    uc = Unitcell("honeycomb", pot.a * 1.1)
    uc.add_site(Site("A", [0, 0], Orbital("s")))
    lattice = Lattice("honeycomb", uc, (True, True))
    lattice.add_unitcell((0, 0), update=True)
    with pytest.raises(ValueError, match="unit vectors"):
        WannierHamiltonianBuilder(pot, 0.5, lattice, verbose=False)


def test_sites_with_several_orbitals_are_rejected():
    pot = honeycomb()
    uc = Unitcell("honeycomb", pot.a)
    uc.add_site(Site("A", [0, 0], [Orbital("px"), Orbital("py")]))
    lattice = Lattice("honeycomb", uc, (True, True))
    lattice.add_unitcell((0, 0), update=True)
    with pytest.raises(ValueError, match="one s-like orbital"):
        WannierHamiltonianBuilder(pot, 0.5, lattice, verbose=False)


def test_building_without_reciprocal_coordinates_says_so(honey):
    with pytest.raises(ValueError, match="reciprocal coordinates"):
        honey.on_lattice().build()
