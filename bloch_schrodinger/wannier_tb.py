"""Tight-binding models derived from a continuous potential, through its maximally localized Wannier functions.

The 'WannierHamiltonianBuilder' class takes the potential of a lattice (a 'Potential' object, with any number of
parameter dimensions) and the tight-binding geometry meant to describe it (a 'tightbinding.Lattice' object), and
returns a 'tightbinding.Hamiltonianbuilder' with every on-site energy and hopping amplitude already registered.
See the 'WannierTightBinding' tutorial for a walk through the pipeline.

This module requires the 'tightbinding' package (https://github.com/GuillotMartin/tightbinding).
"""

import itertools
import re
import warnings

import numpy as np
import xarray as xr

from bloch_schrodinger.fdsolver import FDSolver
from bloch_schrodinger.potential import Potential
from bloch_schrodinger.progress import bar
from bloch_schrodinger.wannier import Wannier

try:
    from tightbinding.geometry import Lattice
    from tightbinding.hamiltonian import Energy, Hamiltonianbuilder, Hopping
except ImportError as error:
    raise ImportError(
        "bloch_schrodinger.wannier_tb builds on the 'tightbinding' package, install it with "
        "'pip install git+https://github.com/GuillotMartin/tightbinding'"
    ) from error


# Sentinel for 'on_lattice': None is a meaningful value of the selection arguments, so it cannot mean "keep
# the current one"
_KEEP = object()

# The magnitude threshold used when no distance criterion is given
_DEFAULT_THRESHOLD = 1e-3

# Names the Hamiltonianbuilder evaluation context already uses
_RESERVED = {"i", "j", "np", "distance"}


def _sanitize(name: str) -> str:
    """Turn an orbital key such as 'A-1_s' into a string usable inside a Python identifier."""
    return re.sub(r"\W", "_", name)


class WannierHamiltonianBuilder(Hamiltonianbuilder):
    """A 'tightbinding.Hamiltonianbuilder' whose couplings are computed from a continuous potential.

    The pipeline runs in the constructor:

    1. The maximally localized Wannier functions (MLWFs) of the band group are computed with the 'Wannier'
       class, one per orbital of the tight-binding unit cell, starting from gaussians centered on the sites.
    2. Each WF is matched to the site it sits on, and translated back into the home unit cell if the
       minimization left it in a neighboring one.
    3. The hopping amplitudes t_ij(R) = <w_i,0|H|w_j,R> are obtained from the Bloch energies, by Fourier
       transforming the Hamiltonian in the Wannier gauge, H_ij(k) = sum_m conj(U_mi(k)) E_m(k) U_mj(k):
       t_ij(R) = 1/N_k sum_k exp(ik.R) H_ij(k). No real-space Hamiltonian is ever built.
    4. The couplings selected, by neighbor shell, bond length or magnitude, are registered on the builder, as
       'Energy' and 'Hopping' terms whose strengths are arrays over the parameter space of the potential.

    The object is then used as any Hamiltonianbuilder: 'build' then 'eigh'. Since every amplitude is stored,
    changing the k-path ('add_parameters') or the lattice ('on_lattice') requires no new computation.
    """

    def __init__(
        self,
        potential: Potential,
        alpha: float,
        lattice: Lattice,
        params: dict | None = None,
        reciprocalcoords: list[str] | None = None,
        k_resolution: int | tuple[int, ...] = 15,
        first_band: int = 0,
        neighbors: int | None = None,
        cutoff: float | None = None,
        threshold: float | None = None,
        solver: str = "pw",
        parallel: bool = False,
        n_cores: int = -1,
        seed: int | None = None,
        verbose: bool = True,
        **solve_kwargs,
    ):
        """Compute the tight-binding parameters of a potential and register them on a Hamiltonianbuilder.

        Args:
            potential (Potential): The potential, defined over a single unit cell. Its parameter dimensions
            become dimensions of the Hamiltonian.
            alpha (float): The kinetic energy coefficient hbar²/2m.
            lattice (Lattice): The tight-binding geometry. Its unit vectors must be those of the potential, each
            of its sites must carry a single orbital, and the site positions must be given in the cartesian
            frame of the potential, as close as possible to the potential minima. Any lattice built from this
            unit cell can be passed: a single periodic cell, a ribbon, a finite flake...
            params (dict, optional): Additional parameters of the Hamiltonian, typically the reciprocal
            coordinates, e.g. {"kx": np.linspace(-1, 1, 100), "ky": 0}. Defaults to None.
            reciprocalcoords (list[str], optional): The names of the reciprocal coordinates. Defaults to
            ["kx", "ky"] (or ["kx"] in 1D) if the lattice has a periodic direction.
            k_resolution (int | tuple[int, ...], optional): The Monkhorst-Pack mesh used for the Wannier
            functions, one value per axis or a single one for all. It sets the range of the hoppings resolved,
            |R_i| <= (n_i - 1)/2 unit cells. Odd values are recommended. Defaults to 15.
            first_band (int, optional): Index of the lowest band of the group. The group spans as many bands
            as the unit cell has orbitals, and must be isolated from the other bands. Defaults to 0.
            neighbors (int, optional): Only keep the hoppings within the first 'neighbors' shells of bond lengths
            of the lattice: 1 for nearest neighbors, 2 for up to next-nearest neighbors, and so on. The shells
            are those of the site positions of the lattice, listed in 'shell_distances'. Defaults to None.
            cutoff (float, optional): Only keep the hoppings shorter than this distance. Defaults to None.
            threshold (float, optional): Only keep the hoppings whose largest magnitude over the parameter space
            is at least 'threshold' times that of the strongest hopping. Defaults to None, which means 1e-3 if
            neither 'neighbors' nor 'cutoff' is given, and no threshold at all if one of them is. All the
            criteria given apply together.
            solver (str, optional): The Bloch solver, 'pw' or 'fd', see 'Wannier'. Defaults to 'pw'.
            parallel (bool, optional): Whether to parallelize the Wannier computation. Defaults to False.
            n_cores (int, optional): The number of cores to use, -1 for all of them. Defaults to -1.
            seed (int, optional): Seed of the randomized initial guess, see 'Wannier.solve'. Defaults to None.
            verbose (bool, optional): Whether to plot progress bars. Defaults to True.
            **solve_kwargs: Passed on to 'Wannier.solve': tol, max_iter, method, blockwargs (e.g.
            blockwargs={'E_lim': 200} to set the plane wave cut-off).

        Raises:
            ValueError: If the lattice and the potential do not describe the same unit cell, if a site carries
            more than one orbital, or if the Wannier functions cannot be matched with the sites.
        """
        self._extract(
            potential, alpha, lattice, k_resolution, first_band, solver,
            parallel, n_cores, seed, verbose, solve_kwargs,
        )
        self._select(neighbors, cutoff, threshold)
        self._register(lattice, params, reciprocalcoords)

    ### ====================================================================
    ### Extraction of the tight-binding parameters
    ### ====================================================================

    def _extract(
        self, potential, alpha, lattice, k_resolution, first_band, solver,
        parallel, n_cores, seed, verbose, solve_kwargs,
    ):
        """Compute the WFs, align them with the sites and Fourier transform the Hamiltonian, see '__init__'."""
        n_dims = potential.n_dims
        if lattice.dimension != n_dims:
            raise ValueError(
                f"The lattice is {lattice.dimension}D but the potential is {n_dims}D"
            )
        if n_dims > 2:
            raise ValueError("Only 1D and 2D lattices are supported, see 'Wannier'")

        if not np.allclose(
            np.asarray(lattice.unitvecs, dtype=float), potential.a,
            rtol=1e-8, atol=1e-10 * np.abs(potential.a).max(),
        ):
            raise ValueError(
                "The lattice and the potential must share the same unit vectors, got "
                f"{np.asarray(lattice.unitvecs).tolist()} and {potential.a.tolist()}"
            )

        unitcell = lattice.unitcell
        for name, site in unitcell.sites.items():
            if site.num_orbitals() != 1:
                raise ValueError(
                    f"Site {name!r} carries {site.num_orbitals()} orbitals: only one s-like orbital per "
                    "site is supported, each orbital being described by a single Wannier function"
                )

        # The orbitals in the order of the unit cell index, which is also the order of the WFs below
        self.orbitals = sorted(unitcell.orbitals_index, key=unitcell.orbitals_index.get)
        self._orbital_names = {
            f"{ksite}_{korb}": (ksite, korb)
            for ksite, site in unitcell.sites.items()
            for korb in site.orbitals
        }
        self.positions = np.array(
            [unitcell.sites[self._orbital_names[orb][0]].position for orb in self.orbitals], dtype=float
        )

        if isinstance(k_resolution, int):
            k_resolution = (k_resolution,) * n_dims
        self.k_resolution = tuple(k_resolution)
        if any(n % 2 == 0 for n in self.k_resolution):
            warnings.warn(
                f"k_resolution = {self.k_resolution} has even entries: the hoppings are then resolved over "
                "a box that misses one row of unit cells, and the band structure is no longer reproduced "
                "exactly on the k-mesh. Prefer odd values.",
                stacklevel=3,
            )

        self.first_band = first_band
        n_orb = len(self.orbitals)
        self.wannier = Wannier(
            potential, alpha, list(2 * np.pi * np.linalg.inv(potential.a).T), self.k_resolution,
            method=solver,
        )
        U = self.wannier.solve(
            (first_band, first_band + n_orb),
            centers=[list(p) for p in self.positions],
            parallel=parallel,
            n_cores=n_cores,
            seed=seed,
            verbose=verbose,
            **solve_kwargs,
        )

        self.param_dims = [
            d for d in U.dims if d not in ["m", "n", *self.wannier.kb_dims]
        ]
        self.U_mnk = self._align(U, verbose)
        self._fourier()

    def _neighbor_distance(self) -> float:
        """The shortest distance between two distinct sites of the infinite lattice."""
        A = self.wannier.potential.a
        dmin = np.inf
        for R in itertools.product(range(-2, 3), repeat=len(A)):
            Rc = np.array(R) @ A
            for i, j in itertools.product(range(len(self.positions)), repeat=2):
                if i == j and not any(R):
                    continue
                dmin = min(dmin, np.linalg.norm(Rc + self.positions[j] - self.positions[i]))
        return dmin

    def _align(self, U: xr.DataArray, verbose: bool) -> xr.DataArray:
        """Match each WF with the site it sits on and bring it back into the home unit cell.

        The minimization only knows the sites through its starting point, so a WF can end up on another
        site than the one it was seeded on, or on a translated copy of it. Neither changes the band
        structure, but both would scramble the labels of the couplings.

        Args:
            U (xr.DataArray): The unitary matrix returned by 'Wannier.solve'.
            verbose (bool): Whether to plot progress bars.

        Raises:
            ValueError: If at some point of the parameter space the WFs cannot be matched one-to-one with
            the sites.

        Returns:
            xr.DataArray: The unitary matrix, with WF n sitting on orbital n of the home unit cell.
        """
        A = self.wannier.potential.a
        kb_dims = self.wannier.kb_dims
        n_orb = len(self.orbitals)
        k = [kc.transpose(*kb_dims).values for kc in self.wannier.k]

        centers = self.wannier.centers(U, verbose=verbose)
        U = U.transpose(*self.param_dims, "m", "n", *kb_dims).copy()
        aligned = xr.DataArray(
            np.zeros(centers.shape),
            coords={
                **{d: centers.coords[d] for d in self.param_dims},
                "orbital": self.orbitals,
                "axis": centers.coords["axis"].values,
            },
            dims=[*self.param_dims, "orbital", "axis"],
        )

        # Candidate translations around the nearest one, which rounding the fractional coordinates
        # does not always find on a skewed lattice
        around = np.array(list(itertools.product((-1, 0, 1), repeat=len(A))))
        residual_max = 0.0
        selections = Wannier._flatten({d: U.coords[d] for d in self.param_dims})
        for sel in selections:
            c = centers.sel(sel).transpose("n", "axis").values
            site_of, shift_of, residual_of = [], [], []
            for n in range(n_orb):
                best = (np.inf, None, None)
                for m in range(n_orb):
                    frac = np.round((c[n] - self.positions[m]) @ np.linalg.inv(A))
                    for R in frac + around:
                        dist = np.linalg.norm(c[n] - self.positions[m] - R @ A)
                        if dist < best[0]:
                            best = (dist, m, R)
                residual_of += [best[0]]
                site_of += [best[1]]
                shift_of += [best[2] @ A]

            if sorted(site_of) != list(range(n_orb)):
                raise ValueError(
                    f"At {sel or 'the single parameter point'}, the Wannier centers "
                    f"{np.round(c, 4).tolist()} cannot be matched one-to-one with the sites "
                    f"{dict(zip(self.orbitals, np.round(self.positions, 4).tolist()))}. Place the "
                    "sites closer to the potential minima."
                )

            U_sel = U.sel(sel).values
            new = np.empty_like(U_sel)
            for m in range(n_orb):
                n = site_of.index(m)
                # Multiplying u_nk by exp(-ik.R) translates w_n by -R
                phase = np.exp(-1j * sum(k[d] * shift_of[n][d] for d in range(len(A))))
                new[:, m] = U_sel[:, n] * phase
                aligned.loc[{**sel, "orbital": self.orbitals[m]}] = c[n] - shift_of[n]
                residual_max = max(residual_max, residual_of[n])
            U.loc[sel] = new

        tolerance = 0.25 * self._neighbor_distance()
        if residual_max > tolerance:
            warnings.warn(
                f"Some Wannier functions sit up to {residual_max:.3g} away from their site (more than a quarter "
                "of the nearest-neighbor distance). The tight-binding spectrum is unaffected, but the site "
                "positions of the lattice do not match the potential minima: check 'centers'.",
                stacklevel=4,
            )

        self.centers = aligned
        return U

    def _fourier(self):
        """Fourier transform the Hamiltonian in the Wannier gauge into the hopping amplitudes."""
        A = self.wannier.potential.a
        kb_dims = self.wannier.kb_dims
        n_dims = len(A)
        self.R_dims = [f"R{i + 1}" for i in range(n_dims)]

        # H_ij(k) = sum_m conj(U_mi) E_m U_mj, the Hamiltonian in the basis of the WFs Bloch sums
        U = self.U_mnk.assign_coords(n=self.orbitals)
        HW = (
            np.conj(U.rename(n="orb1")) * self.wannier.eigva * U.rename(n="orb2")
        ).sum("m")

        # The box of translations resolved by the k-mesh, centered on the home unit cell
        R_coords = {
            dim: np.arange(-((n - 1) // 2), (n - 1) // 2 + 1)
            for dim, n in zip(self.R_dims, self.k_resolution)
        }
        R_int = [xr.DataArray(v, coords={dim: v}) for dim, v in R_coords.items()]
        R_cart = [sum(R_int[i] * A[i, c] for i in range(n_dims)) for c in range(n_dims)]
        phase = np.exp(1j * sum(kc * Rc for kc, Rc in zip(self.wannier.k, R_cart)))
        phase = phase.drop_vars(
            [c for c in phase.coords if c not in [*kb_dims, *self.R_dims]]
        )

        t = xr.dot(HW, phase, dim=kb_dims) / self.wannier.n_k
        # reset_coords drops the scalar coordinates left by the length-1 parameters the solvers squeezed
        t = t.transpose(*self.param_dims, "orb1", "orb2", *self.R_dims).reset_coords(drop=True)
        if np.abs(t.imag).max() <= 1e-6 * np.abs(t).max():
            t = t.real
        self.hoppings = t.rename("hopping")

        positions = xr.DataArray(
            self.positions, coords={"orbital": self.orbitals, "axis": list(range(n_dims))}
        )
        separation = [
            R_cart[c] + positions.sel(axis=c).rename(orbital="orb2")
            - positions.sel(axis=c).rename(orbital="orb1")
            for c in range(n_dims)
        ]
        self.distances = (
            sum(s**2 for s in separation) ** 0.5
        ).drop_vars("axis", errors="ignore").transpose("orb1", "orb2", *self.R_dims).rename("distance")

    ### ====================================================================
    ### Selection and registration of the couplings
    ### ====================================================================

    def _shells(self) -> tuple[np.ndarray, float]:
        """The distinct bond lengths of the lattice whose bonds are all within the resolved translations.

        Returns:
            tuple[np.ndarray, float]: The sorted shell distances, and the tolerance used to tell two lengths apart.
        """
        A = self.wannier.potential.a
        tolerance = 1e-6 * np.abs(A).max()
        half = [(n - 1) // 2 for n in self.k_resolution]

        # The shortest bond to a translation just outside the box: from there on, a shell may miss members
        radius = np.inf
        for R in itertools.product(*[range(-h - 1, h + 2) for h in half]):
            if all(abs(r) <= h for r, h in zip(R, half)):
                continue
            Rc = np.array(R) @ A
            for i, j in itertools.product(range(len(self.positions)), repeat=2):
                radius = min(radius, np.linalg.norm(Rc + self.positions[j] - self.positions[i]))

        lengths = self.distances.values.ravel()
        lengths = np.sort(lengths[(lengths > tolerance) & (lengths < radius - tolerance)])
        if lengths.size == 0:
            return lengths, tolerance
        return lengths[np.concatenate([[True], np.diff(lengths) > tolerance])], tolerance

    def _select(self, neighbors: int | None, cutoff: float | None, threshold: float | None):
        """Decide which couplings are kept and gather them in 'tb_parameters', see '__init__'."""
        # The values as given, None included, so that 'on_lattice' can tell what was asked for
        self.neighbors = neighbors
        self.cutoff = cutoff
        self.threshold = threshold

        self.shell_distances, tolerance = self._shells()
        if threshold is None:
            threshold = 0.0 if neighbors is not None or cutoff is not None else _DEFAULT_THRESHOLD
        self._effective_threshold = threshold

        max_distance = np.inf
        if neighbors is not None:
            if neighbors < 1:
                raise ValueError(f"'neighbors' must be a positive integer, got {neighbors}")
            if neighbors > len(self.shell_distances):
                raise ValueError(
                    f"The {'x'.join(map(str, self.k_resolution))} k-mesh only resolves the first "
                    f"{len(self.shell_distances)} shells of neighbors entirely, increase k_resolution to reach "
                    f"{neighbors} shells"
                )
            max_distance = self.shell_distances[neighbors - 1]
        if cutoff is not None:
            max_distance = min(max_distance, cutoff)

        def shell_of(distance: float) -> int | None:
            index = int(np.searchsorted(self.shell_distances, distance - tolerance))
            if index < len(self.shell_distances) and abs(self.shell_distances[index] - distance) <= tolerance:
                return index + 1
            return None

        amplitude = abs(self.hoppings)
        if self.param_dims:
            amplitude = amplitude.max(self.param_dims)
        R_box = list(
            itertools.product(*[self.hoppings.coords[d].values.tolist() for d in self.R_dims])
        )
        zero = (0,) * len(self.R_dims)
        offsite = [
            float(amplitude.sel(orb1=o1, orb2=o2, **dict(zip(self.R_dims, R))))
            for o1 in self.orbitals for o2 in self.orbitals for R in R_box
            if not (o1 == o2 and R == zero)
        ]
        reference = max(offsite) if offsite else 0.0

        self.coupling_table = []
        variables = {}
        for i, o1 in enumerate(self.orbitals):
            # On-site energies, real since H is hermitian
            name = f"E_{_sanitize(o1)}"
            value = self.hoppings.sel(orb1=o1, orb2=o1, **dict(zip(self.R_dims, zero)))
            variables[name] = value.real.drop_vars(["orb1", "orb2", *self.R_dims])
            self.coupling_table += [
                {"name": name, "orb1": o1, "orb2": o1, "displacement": zero, "distance": 0.0, "shell": 0,
                     "amplitude": float(abs(value).max())}
            ]

            for o2 in self.orbitals[i:]:
                for R in R_box:
                    # Each bond is registered once, the builder adds its hermitian conjugate
                    if o1 == o2 and R <= zero:
                        continue
                    where = dict(orb1=o1, orb2=o2, **dict(zip(self.R_dims, R)))
                    distance = float(self.distances.sel(where))
                    if distance > max_distance + tolerance:
                        continue
                    if float(amplitude.sel(where)) < threshold * reference:
                        continue
                    displacement = "_".join(str(r).replace("-", "m") for r in R)
                    name = f"t_{_sanitize(o1)}_{_sanitize(o2)}_{displacement}"
                    variables[name] = self.hoppings.sel(where).drop_vars(
                        ["orb1", "orb2", *self.R_dims]
                    )
                    self.coupling_table += [
                        dict(name=name, orb1=o1, orb2=o2, displacement=R, distance=distance,
                             shell=shell_of(distance), amplitude=float(amplitude.sel(where)))
                    ]

        names = [c["name"] for c in self.coupling_table]
        if len(set(names)) != len(names):
            raise ValueError(
                f"The orbital names {self.orbitals} give clashing parameter names once sanitized, rename them"
            )

        self.tb_parameters = xr.Dataset(variables)
        for c in self.coupling_table:
            attrs = dict(orb1=c["orb1"], orb2=c["orb2"], displacement=c["displacement"], distance=c["distance"])
            if c["shell"] is not None:
                attrs["shell"] = c["shell"]
            self.tb_parameters[c["name"]].attrs.update(attrs)

    def _check_lattice(self, lattice: Lattice):
        """Check that a lattice is built from the unit cell the parameters were computed for."""
        if not np.allclose(np.asarray(lattice.unitvecs, dtype=float), self.wannier.potential.a):
            raise ValueError("The lattice does not share the unit vectors of the potential")
        orbitals = sorted(lattice.unitcell.orbitals_index, key=lattice.unitcell.orbitals_index.get)
        if sorted(orbitals) != sorted(self.orbitals):
            raise ValueError(
                f"The lattice orbitals {orbitals} differ from those the parameters were computed for, "
                f"{self.orbitals}"
            )
        for orb, position in zip(self.orbitals, self.positions):
            site = lattice.unitcell.sites[self._orbital_names[orb][0]]
            if not np.allclose(site.position, position):
                raise ValueError(f"The site of orbital {orb!r} has moved")
        if len(lattice.unitcells) == 0:
            raise ValueError("The lattice has no unit cells, add some with 'add_unitcell'")

    def _register(self, lattice: Lattice, params: dict | None, reciprocalcoords: list[str] | None):
        """Initialize the Hamiltonianbuilder and register every coupling kept by '_select'."""
        self._check_lattice(lattice)
        params = dict(params or {})
        self._user_params = params

        clash = set(params) & set(self.param_dims)
        if clash:
            raise ValueError(
                f"{sorted(clash)} are parameter dimensions of the potential, they cannot be passed in 'params'"
            )
        clash = set(self.tb_parameters.data_vars) & (set(params) | _RESERVED)
        if clash:
            raise ValueError(f"{sorted(clash)} are reserved for the tight-binding parameters")

        if reciprocalcoords is None:
            reciprocalcoords = (
                ["kx", "ky", "kz"][: lattice.dimension] if any(lattice.periodicity) else []
            )

        all_params = {d: self.hoppings.coords[d].values for d in self.param_dims}
        all_params.update(params)

        # Read by 'create_array', which the base constructor calls
        self._tb_values = {
            name: var.transpose(*self.param_dims) for name, var in self.tb_parameters.data_vars.items()
        }
        super().__init__(lattice, all_params, reciprocalcoords)

        onsite = Energy(lattice)
        couplings = []
        for c in self.coupling_table:
            if c["orb1"] == c["orb2"] and not any(c["displacement"]):
                site, orbital = self._orbital_names[c["orb1"]]
                onsite.set_energy(c["name"], norbital=orbital, nsite=site)
            else:
                couplings += [Hopping(lattice, c["name"], c["orb1"], c["orb2"], c["displacement"])]
        self.set_on_site_energies(onsite)
        self.add_couplings(couplings)

    ### ====================================================================
    ### User-facing methods
    ### ====================================================================

    def create_array(self):
        """Build the (empty) Hamiltonian array and its evaluation context, see 'Hamiltonianbuilder.create_array'.

        The tight-binding parameters are put straight into the context rather than registered with
        'add_multidimensional_coord': a coordinate is handed to the expressions together with every other
        coordinate sharing its dimensions, which with hundreds of couplings makes each evaluation drag
        hundreds of coordinates along, and the build quadratic in their number.
        """
        super().create_array()
        for name, var in getattr(self, "_tb_values", {}).items():
            if var.dims:
                self.context[name] = xr.DataArray(
                    var.values, coords={d: self.Hamiltonian.coords[d].values for d in var.dims}, dims=var.dims
                )
            else:
                self.context[name] = var.values.item()

    def build(self):
        """Fill the Hamiltonian matrix, see 'Hamiltonianbuilder.build'.

        Raises:
            ValueError: If the lattice is periodic but the reciprocal coordinates were never given values.
        """
        if self.periodicedges > 0:
            missing = [
                k for k in self.recoords
                if k not in self.parameters and k not in self.multidim_coords
            ]
            if missing:
                raise ValueError(
                    f"The reciprocal coordinates {missing} have no values: pass them in 'params', or "
                    "with 'add_parameters' or 'add_multidimensional_coord', before building"
                )
        super().build()

    def on_lattice(
        self,
        lattice: Lattice | None = None,
        params: dict | None = None,
        reciprocalcoords: list[str] | None = None,
        neighbors: int | None = _KEEP,
        cutoff: float | None = _KEEP,
        threshold: float | None = _KEEP,
    ) -> "WannierHamiltonianBuilder":
        """Return a new builder sharing the tight-binding parameters of this one, without recomputing them.

        Useful to use the couplings on another lattice built from the same unit cell (a ribbon, a flake, a
        supercell...), or to change the selection of the couplings.

        Args:
            lattice (Lattice, optional): The new lattice. Defaults to the current one.
            params (dict, optional): The additional parameters of the Hamiltonian. Defaults to those given to
            this builder.
            reciprocalcoords (list[str], optional): The names of the reciprocal coordinates. Defaults to those of
            this builder if the lattice is unchanged, to the defaults of the constructor otherwise.
            neighbors (int, optional): See '__init__'. Defaults to the current value, unless 'cutoff' is given.
            cutoff (float, optional): See '__init__'. Defaults to the current value, unless 'neighbors' is given.
            Passing one of the two distance criteria replaces the selection by distance altogether: the other
            one is then reset to None rather than kept, so that it does not silently clip the new selection.
            threshold (float, optional): See '__init__'. Defaults to the current value.

        Returns:
            WannierHamiltonianBuilder: The new builder, not yet built.
        """
        # Creating the new instance without recomputing teh wannier functions.
        new = object.__new__(type(self))
        for attr in [ # Any new attribute created my _extract must be added here
            "wannier", "orbitals", "_orbital_names", "positions", "k_resolution", "first_band",
            "param_dims", "U_mnk", "centers", "R_dims", "hoppings", "distances",
        ]:
            setattr(new, attr, getattr(self, attr))

        if neighbors is _KEEP and cutoff is _KEEP:
            neighbors, cutoff = self.neighbors, self.cutoff
        else:
            neighbors = None if neighbors is _KEEP else neighbors
            cutoff = None if cutoff is _KEEP else cutoff
        new._select(neighbors, cutoff, self.threshold if threshold is _KEEP else threshold)
        if reciprocalcoords is None and lattice is None:
            reciprocalcoords = self.recoords
        new._register(
            self.lattice if lattice is None else lattice,
            self._user_params if params is None else params,
            reciprocalcoords,
        )
        return new

    def real_space_hoppings(
        self,
        bounds: list[tuple[int, int]] | None = None,
        coarsen: tuple[int, ...] | None = None,
        verbose: bool = True,
    ) -> xr.Dataset:
        """Recompute the registered parameters as real-space integrals <w_i,0|H|w_j,R>, as an independent check.

        The WFs are built over a tiled region with 'Wannier.compute_wannier', and H is the finite-difference
        Hamiltonian of the tiled potential. It is much slower than the k-space route of the constructor, and
        carries the discretization error of the finite differences.

        Args:
            bounds (list[tuple[int, int]], optional): The range of unit cells to tile along each lattice vector.
            Defaults to two cells beyond the longest registered hopping on each side.
            coarsen (tuple[int, ...], optional): Coarsening factors of the grid, see 'Wannier.compute_wannier'.
            Defaults to no coarsening.
            verbose (bool, optional): Whether to plot progress bars. Defaults to True.

        Returns:
            xr.Dataset: The same variables as 'tb_parameters'.
        """
        n_dims = len(self.R_dims)
        if bounds is None:
            reach = [
                max(abs(c["displacement"][i]) for c in self.coupling_table) for i in range(n_dims)
            ]
            bounds = [(-r - 2, r + 3) for r in reach]
        if coarsen is None:
            coarsen = (1,) * n_dims

        tiled, wannier = self.wannier.compute_wannier(
            self.U_mnk, bounds, coarsen=coarsen, verbose=verbose
        )
        # The Bloch solvers squeeze the length-1 parameter dimensions of the potential away, and so are
        # they from the amplitudes: the tiled potential must follow, or its selections would not match
        tiled.V = tiled.V.squeeze(drop=True)
        wannier = wannier.assign_coords(n=self.orbitals)
        n_cell = [
            self.wannier.potential.resolution[i] // coarsen[i] for i in range(n_dims)
        ]
        spatial_dims = self.wannier.spatial_dims

        solver = FDSolver(tiled, self.wannier.alpha)
        # create_hamiltonian takes one selection per type of parameter
        types = ["potential", "alpha", "reciprocal", "coupling"]
        dims = {t: [d for d, coord in solver.allcoords.items() if coord[0] == t] for t in types}
        dS = tiled.get_dS()

        values = {
            name: np.zeros([len(self.hoppings.coords[d]) for d in self.param_dims], dtype=complex)
            for name in self.tb_parameters.data_vars
        }
        selections = Wannier._flatten({d: self.hoppings.coords[d] for d in self.param_dims})
        for sel in bar(selections, desc="Computing matrix elements", unit="point", verbose=verbose):
            index = tuple(
                int(np.argmax(self.hoppings.coords[d].values == sel[d])) for d in self.param_dims
            )
            H0 = solver.create_hamiltonian(*[{d: sel[d] for d in dims[t]} for t in types])
            w = wannier.sel(sel).transpose("n", *spatial_dims)
            H_w = {
                orb: (H0 @ w.sel(n=orb).values.reshape(-1)).reshape(w.sel(n=orb).shape)
                for orb in self.orbitals
            }
            for c in self.coupling_table:
                # <w_i,0|H|w_j,R> = <w_i,-R|H|w_j,0>, the tile being periodic
                w_bra = np.roll(
                    w.sel(n=c["orb1"]).values,
                    [-r * n for r, n in zip(c["displacement"], n_cell)],
                    axis=list(range(n_dims)),
                )
                values[c["name"]][index] = np.vdot(w_bra, H_w[c["orb2"]]) * dS

        coords = {d: self.hoppings.coords[d].values for d in self.param_dims}
        result = xr.Dataset(
            {name: (self.param_dims, v) for name, v in values.items()}, coords=coords
        )
        if all(np.abs(v.imag).max() <= 1e-6 * np.abs(v).max() for v in values.values()):
            result = result.map(lambda v: v.real)
        return result

    def __repr__(self) -> str:
        shape = {d: len(self.hoppings.coords[d]) for d in self.param_dims}
        bands = f"{self.first_band}-{self.first_band + len(self.orbitals) - 1}"
        neighbors = (
            f"{self.neighbors} (<= {self.shell_distances[self.neighbors - 1]:.4g})"
            if self.neighbors is not None else "None"
        )
        threshold = f"{self._effective_threshold:g}" if self._effective_threshold > 0 else "off"
        lines = [
            f"WannierHamiltonianBuilder on lattice {self.lattice.name!r}: orbitals {self.orbitals}, "
            f"bands {bands}, k-mesh {'x'.join(map(str, self.k_resolution))}, parameters {shape}",
            f"{len(self.coupling_table)} terms (neighbors = {neighbors}, cutoff = {self.cutoff}, "
            f"threshold = {threshold}):",
            f"  {'name':<24} {'orbitals':<16} {'displacement':<14} {'shell':>5} {'distance':>10} {'max |value|':>12}",
        ]
        for c in self.coupling_table:
            shell = "-" if c["shell"] is None else str(c["shell"])
            lines += [
                f"  {c['name']:<24} {c['orb1'] + ' -> ' + c['orb2']:<16} {c['displacement']!s:<14} "
                f"{shell:>5} {c['distance']:>10.4g} {c['amplitude']:>12.4g}"
            ]
        return "\n".join(lines)
