import itertools
import warnings
from typing import Union

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from numpy.linalg import inv, svd
from numpy.random import uniform
from scipy.linalg import expm, fractional_matrix_power
from tqdm import tqdm
from xarray_einstats.linalg import matmul

from bloch_schrodinger.fdsolver import FDSolver
from bloch_schrodinger.potential import Potential, create_parameter
from bloch_schrodinger.pwsolver import PWSolver


def trH(arr: xr.DataArray) -> xr.DataArray:
    cop = arr.copy()
    cop.data = np.conjugate(
        np.moveaxis(
            cop.data,
            source=[arr.get_axis_num("m"), arr.get_axis_num("n")],
            destination=[arr.get_axis_num("n"), arr.get_axis_num("m")],
        )
    )
    return cop

def xexpm(
    arr: xr.DataArray,
) -> xr.DataArray:  # A function to compute the matrix exponential
    return xr.apply_ufunc(
        expm, arr, input_core_dims=[["m", "n"]], output_core_dims=[["m", "n"]]
    )

class Wannier:
    """The Wannier class uses the algorithm developped by Marzari and Vanderbild (10.1103/PhysRevB.56.12847) 
    to determine the Maximally Localized Wannier Functions (MLWFs) of a given lattice.
    """
    
    def __init__(
        self,
        Potential: Potential,
        alpha: Union[float, xr.DataArray],
        rec_vecs: list[list[float, float]],
        resolution: tuple[int, int],
        method: str = 'pw'
        ):
        """Initialize a Wannier object. This class allows the determination of the MLWFs for a whole parameter space.

        Args:
            Potential (Potential): The potential to use for the Bloch function computation
            alpha (Union[float, xr.DataArray]): The kinetic term
            rec_vecs (list[list[float]]): The reciprocal vectors of the unit cell, one per axis, given as
            [b1, ...] with bi a list or array of length n_dims.
            resolution (tuple[int]): The resolution for the k-space Monkhorst-Pack grid used, one per axis.
            method (str, optional): The Bloch-Schrödinger solver to use. Either 'pw' for plane waves or 'fd' for finite differences. Defaults to 'pw'.

        Raises:
            ValueError: If the potential is 3D, which this class does not support, or if rec_vecs and
            resolution do not match the potential's dimensionality.
        """
        self.n_dims = Potential.n_dims
        if self.n_dims > 2:
            raise ValueError(
                f"Wannier supports 1D and 2D potentials, got a {self.n_dims}D one. The rest of the "
                "package (Potential, FDSolver, PWSolver and the plotting functions) handles 3D, but "
                "the Marzari-Vanderbilt machinery here does not: the spread functional's finite "
                "difference stencil is solved on shells of k-points, and neither that nor the Wannier "
                "profile reconstruction has been worked out for a 3D Brillouin zone."
            )
        if len(rec_vecs) != self.n_dims or len(resolution) != self.n_dims:
            raise ValueError(
                f"For a {self.n_dims}D potential, Wannier needs {self.n_dims} reciprocal vector(s) and "
                f"a length-{self.n_dims} resolution, got {len(rec_vecs)} and {len(resolution)}"
            )

        self.potential = Potential
        self.alpha = alpha
        self.method = method

        self.spatial_dims = [f"a{i + 1}" for i in range(self.n_dims)]
        self.kb_dims = [f"kb{i + 1}" for i in range(self.n_dims)]
        self.coord_names = ["x", "y", "z"][: self.n_dims]

        self.b = [np.asarray(v, dtype=float) for v in rec_vecs]
        self.nb = list(resolution)
        self.n_k = int(np.prod(self.nb))  # Total number of k-points in the Monkhorst-Pack grid

        self.kb = [
            create_parameter(
                name, np.linspace(-1 / 2, 1 / 2, n, endpoint=False) + 1 / 2 / n
            )
            for name, n in zip(self.kb_dims, self.nb)
        ]

        # Cartesian components of every k-point, k_c = sum_i b[i][c] * kb[i]
        self.k = [
            sum(self.b[i][c] * self.kb[i] for i in range(self.n_dims))
            for c in range(self.n_dims)
        ]

        self.maxsearch = min(min(self.nb) // 2 - 2, 10)
        self.compute_stencil()


    def compute_stencil(self):
        """Creates the weights for the gradient and laplacian operator definitions, see https://arxiv.org/pdf/0708.0650 sec. 3.2 for more details."""

        centre = [n // 2 for n in self.nb]
        k_grid = [kc.transpose(*self.kb_dims) for kc in self.k]
        k0 = [float(kc[tuple(centre)]) for kc in k_grid]

        lim = self.maxsearch
        # Computing the distances in index and cartesian coordinates for a few shells
        offsets, deltas, Dist = [], [], []
        for off in itertools.product(range(-lim, lim + 1), repeat=self.n_dims):
            if all(o == 0 for o in off):
                continue
            idx = tuple(centre[d] + off[d] for d in range(self.n_dims))
            delta = [float(k_grid[c][idx]) - k0[c] for c in range(self.n_dims)]
            offsets += [off]
            deltas += [delta]
            Dist += [sum(d**2 for d in delta) ** 0.5]

        offsets = np.array(offsets)
        deltas = np.array(deltas)
        Dist = np.array(Dist)

        # Sorting all the points by ascending distance to the central one
        sorting = np.argsort(Dist)
        offsets_sorted = offsets[sorting]
        deltas_sorted = deltas[sorting]
        Dist_sorted = Dist[sorting]

        # Sorting the points by shell number
        shell_distance, start_shell, shell_index, n_in_shell = np.unique(
            Dist_sorted.round(decimals=8),
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )

        # The condition sum_b w_b b_alpha b_beta = delta_alpha_beta, one row per independent pair of
        # cartesian directions: (xx) in 1D, (xx, xy, yy) in 2D
        pairs = [
            (c1, c2) for c1 in range(self.n_dims) for c2 in range(c1, self.n_dims)
        ]
        q = np.array([1.0 if c1 == c2 else 0.0 for c1, c2 in pairs])

        # Solve for the weights, keep adding shells until the condition is met
        solved = False
        nshell = 1
        while not solved:
            A_js = np.array(
                [
                    [
                        sum(
                            deltas_sorted[start_shell[s] + m][c1]
                            * deltas_sorted[start_shell[s] + m][c2]
                            for m in range(n_in_shell[s])
                        )
                        for s in range(nshell)
                    ]
                    for c1, c2 in pairs
                ],
            )

            U, Sdiag, Vh = svd(A_js, full_matrices=False)
            S = np.diag(Sdiag)

            V = np.transpose(np.conjugate(Vh))
            Uh = np.transpose(np.conjugate(U))

            w = V @ inv(S) @ Uh @ q
            if np.all(np.isclose(A_js @ w, q)):
                solved = True
                self.n_shell = nshell
            else:
                nshell += 1
                if nshell > len(n_in_shell):
                    raise ValueError(
                        "Could not satisfy the stencil condition on this k-mesh. Increase the "
                        "Monkhorst-Pack resolution."
                    )

        weights = []
        neighbors = []
        neighbors_indexes = []
        for s in range(nshell):
            for m in range(n_in_shell[s]):
                indx = start_shell[s] + m
                neighbors_indexes += [list(offsets_sorted[indx])]
                neighbors += [list(deltas_sorted[indx])]
                weights += [w[s]]

        self.stencil_size = len(weights)
        self.weights = xr.DataArray(weights, coords={"b": np.arange(self.stencil_size)})
        # 'kxy' indexes the cartesian components of each stencil vector b
        self.neighbors = xr.DataArray(
            neighbors,
            coords={
                "b": np.arange(self.stencil_size),
                "kxy": np.arange(self.n_dims),
            },
        )
        self.neighbors_indexes = xr.DataArray(
            neighbors_indexes,
            coords={"b": np.arange(self.stencil_size), "ij": np.arange(self.n_dims)},
        )
        
        
    ### ====================================================================
    ### MLWFs helper functions, see dedicated tutorial for more informations
    ### ====================================================================
    
    def M_mnkb(self, u_mk:xr.DataArray)->xr.DataArray:
        """Compute the overlap matrix M_mnkb = <u_m,k|u_n,k+b>

        Args:
            u_mk (xr.DataArray): The bloch eigenvector array

        Returns:
            xr.DataArray: M_mnkb
        """
        nb = self.stencil_size # shorter names

        M_mnkb = xr.DataArray(
            np.zeros((self.nbands, self.nbands, *self.nb, nb), dtype=np.complex128),
            dims=['m','n',*self.kb_dims,'b']
        )

        for dim, coord in zip(self.kb_dims, self.kb):
            M_mnkb.coords[dim] = coord

        for b_idx in range(nb):
            delta = [
                int(self.neighbors_indexes.sel(b=b_idx, ij=i)) for i in range(self.n_dims)
            ]

            u_nkpb = u_mk.roll(
                {dim: -d for dim, d in zip(self.kb_dims, delta)}
            ).rename({'m':'n'})

            # Rolling wraps k-points around the zone: those get shifted by a reciprocal lattice
            # vector G, which the periodic part of the Bloch function has to be corrected for.
            wraps = []
            for i, (dim, n) in enumerate(zip(self.kb_dims, self.nb)):
                idx = np.arange(n)
                wraps += [
                    xr.DataArray(
                        np.where(idx + delta[i] < 0, -1, np.where(idx + delta[i] >= n, 1, 0)),
                        coords={dim: u_mk.coords[dim].values},
                        dims=dim,
                    )
                ]

            phase_arg = 0
            for c in range(self.n_dims):
                G_c = sum(wraps[i] * self.b[i][c] for i in range(self.n_dims))
                phase_arg = phase_arg + G_c * u_mk.coords[self.coord_names[c]]
            u_nkpb = u_nkpb * np.exp(1j * phase_arg)

            M_mnkb.loc[{'b':b_idx}] = (
                (u_mk.conjugate() * u_nkpb)
                .sum(dim=self.spatial_dims)
                .transpose('m', 'n', *self.kb_dims)
            )

        return M_mnkb * self.potential.get_dS()

    def r_n(self, M_mnkb: xr.DataArray) -> xr.DataArray:
        M_nnkb = M_mnkb.sel(m=M_mnkb.n)
        r = -(self.weights * self.neighbors * xr.ufuncs.angle(M_nnkb)).sum('b').sum(self.kb_dims) / self.n_k
        return r

    def Omega(self, M_mnkb: xr.DataArray) -> xr.DataArray:
        M_nnkb = M_mnkb.sel(m=M_mnkb.n)
        b_dot_rn = (self.neighbors * self.r_n(M_mnkb)).sum("kxy")
        diag = ((xr.ufuncs.angle(M_nnkb) + b_dot_rn) ** 2).sum("n")

        nondiag = (abs(M_mnkb) ** 2).where(M_mnkb.m != M_mnkb.n).sum(["m", "n"])

        omega = ((diag + nondiag)*self.weights).sum('b').sum(self.kb_dims) / self.n_k

        return omega.item()
    
    def q_nkb(self, M_mnkb) -> xr.DataArray:
        M_nnkb = M_mnkb.sel(m=M_mnkb.n)
        return xr.ufuncs.angle(M_nnkb) + (self.neighbors * self.r_n(M_mnkb)).sum("kxy")

    def R_mnkb(self, M_mnkb) -> xr.DataArray:
        M_nnkb = M_mnkb.sel(m=M_mnkb.n)
        return M_mnkb * M_nnkb.conjugate()

    def T_mnkb(self, M_mnkb) -> xr.DataArray:
        M_nnkb = M_mnkb.sel(m=M_mnkb.n)
        return M_mnkb / M_nnkb * self.q_nkb(M_mnkb)

    def G_nmk(self, M_mnkb):
        R = self.R_mnkb(M_mnkb)
        T = self.T_mnkb(M_mnkb)
        return 4 * (self.weights * ((R - trH(R)) / 2 - (T + trH(T)) / 2 / 1j)).sum("b")
    
    def new_M(
        self,
        U: xr.DataArray, 
        M: xr.DataArray
    ) -> xr.DataArray:
        nM = M.copy()
        for ib in range(self.stencil_size):
            U_kpb = U.roll(
                {
                    dim: -int(self.neighbors_indexes[ib, i])
                    for i, dim in enumerate(self.kb_dims)
                }
            )
            nM[{"b": ib}] = matmul(
                trH(U), matmul(M[{"b": ib}], U_kpb, ["m", "n"]), ["m", "n"]
            )

        return nM

    def guess(
        self,
        u_mk: xr.DataArray,
        centers: list[list[float]],
        rng: np.random.Generator | None = None,
    ) -> xr.DataArray:
        """Initialize a matrix U_mnk0 by using gaussian functions.

        Args:
            u_mk (xr.DataArray): The initial eigenvector set
            centers (list[list[float]]): The center of each wannier function, one [x, ...] per function
            rng (np.random.Generator, optional): Source of the random spread given to each trial gaussian.
            Pass a seeded generator to make the whole minimization reproducible; without one the starting
            point, and therefore the minimum reached, differs from run to run. Defaults to None.

        Returns:
            xr.DataArray
        """
        # Taking random points for the centers
        draw = (rng.uniform if rng is not None else uniform)

        sigma = self.potential.a[0]@self.potential.a[0] / 10 # A reasonable spread

        g_n = xr.DataArray(
            np.zeros((self.nbands, *[u_mk.sizes[d] for d in self.spatial_dims])),
            coords = {
                'n': np.arange(self.n_wannier[0], self.n_wannier[1]),
                **{d: u_mk.coords[d] for d in self.spatial_dims},
            }
        )

        for i in range(self.nbands):
            r2 = sum(
                (u_mk.coords[name] - centers[i][c]) ** 2
                for c, name in enumerate(self.coord_names)
            )
            gauss = np.exp(-r2 / 2 / (sigma*draw(0.5, 2))**2)
            gauss /= (gauss**2).sum(self.spatial_dims)
            g_n[{'n':i}] = gauss.transpose(*self.spatial_dims)

        # Now performing a lödwin decomposition
        A_mnk = (u_mk.conjugate() * g_n).sum(self.spatial_dims)
        S_mnk = matmul(trH(A_mnk), A_mnk, ["m", "n"])

        invsqrt_S_mnk = xr.apply_ufunc(
            lambda m: fractional_matrix_power(m, -1 / 2),
            S_mnk,
            input_core_dims=[["m", "n"]],
            output_core_dims=[["m", "n"]],
        )

        U_mnk0 = matmul(A_mnk, invsqrt_S_mnk, ["m", "n"]).transpose('m', 'n', *self.kb_dims)
        
        return U_mnk0
                    
    def compute_bloch(self, **kwargs)->xr.DataArray:
        """Compute the Bloch eigenvectors necessary to the determination of the MLWFs.

        Args:
            n_wannier (_type_): The number of Wannier functions to compute.

        Returns:
            xr.DataArray: The bloch eigenvectors
        """
        
        if self.method == 'pw':
            solv = PWSolver(
                self.potential, self.alpha, **kwargs
            )

            solv.set_reciprocal_space(self.k)
            eigva, eigve = solv.solve(self.n_wannier[1], parallel=True, n_cores = -1)
            eigve = solv.compute_u(eigve)
        
        elif self.method == 'fd':
            solv = FDSolver(self.potential, self.alpha)
            solv.set_reciprocal_space(self.k)
            eigva, eigve = solv.solve(self.n_wannier[1], parallel=True, n_cores = -1)

        else:
            raise ValueError("Method must either be 'pw' or 'fd'")
        
        if self.n_wannier[1] == 1:
            eigve = eigve.expand_dims('band')
        
        self.eigve = eigve.transpose(..., 'band', *self.kb_dims, *self.spatial_dims).rename({'band':'m'})[{'m':slice(self.n_wannier[0], None)}]

    @staticmethod
    def _inner(A: xr.DataArray, B: xr.DataArray) -> float:
        """The real inner product Re<A,B> on the gradient space, summed over k and matrix entries.

        Dropped to numpy on purpose: three of these run per conjugate gradient iteration, and xarray's
        alignment machinery costs more than the arithmetic on arrays this small, enough to cancel out
        the iterations the better search direction saves.
        """
        if A.dims != B.dims:
            B = B.transpose(*A.dims)
        return float(np.real(np.vdot(A.values, B.values)))

    def initial_step(self) -> float:
        """The step length 1/(4.sum_b w_b) that Marzari and Vanderbilt derive for steepest descent.

        It is the exact minimizer along the gradient for the diagonal part of the functional, so it is
        a far better starting guess than an arbitrary constant, and the line search then only has to
        correct for the rest.

        Returns:
            float
        """
        return 1 / (4 * float(self.weights.sum()))

    def compute_U_mnk(
        self,
        sel: dict,
        centers: list[list[float]],
        tol: float,
        max_iter: int = 200,
        method: str = "cg",
        rng: np.random.Generator | None = None,
        return_info: bool = False,
    ) -> xr.DataArray:
        """Determine the MLWFs for a given collection of eigenvectors u_mk at a given parameter space point 'sel',
        by minimizing the spread functional.

        The search direction is either plain steepest descent or, by default, a Polak-Ribiere conjugate
        gradient built on top of it. The step length starts from the analytic value of 'initial_step'
        and is then grown on success and halved on rejection.

        Args:
            sel (xr.DataArray): The point in parameter space for which to find the MLWFs.
            centers (list[list[float]]): The center of each wannier function, one [x, ...] per function.
            tol (float): Convergence threshold on the *relative* decrease of the spread. Being relative,
            it means the same thing whatever the magnitude of the spread happens to be for a given lattice.
            max_iter (int, optional): Hard cap on the number of iterations. Defaults to 200.
            method (str, optional): 'cg' for conjugate gradient, 'sd' for plain steepest descent. Defaults to 'cg'.
            rng (np.random.Generator, optional): Passed to 'guess', see there. Defaults to None.
            return_info (bool, optional): Also return a dict describing how the minimization went.
            Defaults to False.

        Returns:
            xr.DataArray: The unitary matrix transformation U_mnk required to determine the MLWFs, and
            optionally a dict with the iteration count, the final spread and whether it converged.

        Raises:
            ValueError: If method is neither 'cg' nor 'sd'.
        """
        if method not in ("cg", "sd"):
            raise ValueError(f"method must be 'cg' or 'sd', got {method!r}")

        u_mk = self.eigve.sel(sel)
        U0 = self.guess(u_mk, centers, rng=rng)

        M_init = self.M_mnkb(u_mk)
        M0 = self.new_M(U0, M_init)
        Omega0 = self.Omega(M0)  # Initial value of the functional

        alpha = self.initial_step()
        direction = None
        previous_gradient = None
        n_up = 0
        converged = False
        iteration = 0

        while iteration < max_iter:
            iteration += 1
            G0 = self.G_nmk(M0)

            if method == "cg" and previous_gradient is not None:
                # Polak-Ribiere. Clamping beta at zero restarts the recursion along the plain
                # gradient whenever conjugacy stops paying, which is the usual safeguard.
                denominator = self._inner(previous_gradient, previous_gradient)
                beta = (
                    max(
                        (self._inner(G0, G0) - self._inner(G0, previous_gradient))
                        / denominator,
                        0.0,
                    )
                    if denominator > 0
                    else 0.0
                )
                direction = G0 + beta * direction
            else:
                direction = G0
            previous_gradient = G0

            U_trial = matmul(U0, xexpm(alpha * direction), ["m", "n"])
            M_trial = self.new_M(U_trial, M_init)
            Omega_trial = self.Omega(M_trial)

            epsilon = Omega0 - Omega_trial
            if epsilon > 0:
                n_up = 0
                M0 = M_trial.copy()
                U0 = U_trial.copy()
                alpha *= 1.2
                # Stop once the spread has stopped moving relative to its own size, rather than after
                # a fixed run of rejected steps: that tail costs ten full evaluations for nothing.
                converged = epsilon <= tol * max(abs(Omega0), 1e-30)
                Omega0 = Omega_trial
                if converged:
                    break
            else:
                n_up += 1
                alpha *= 0.5
                # A rejected step means the direction was not usable, so the conjugacy is stale
                previous_gradient = None
                if n_up >= 10:
                    converged = True
                    break

        if not converged:
            warnings.warn(
                f"The spread minimization at {sel or 'the single parameter point'} used all "
                f"{max_iter} iterations without meeting tol={tol:g}; the last spread was "
                f"{Omega0:.6g}. Raise max_iter, or loosen tol.",
                stacklevel=2,
            )

        if return_info:
            return U0, {
                "iterations": iteration,
                "spread": Omega0,
                "converged": converged,
            }
        return U0


    def solve(
        self, 
        n_wannier: Union[int, tuple[int, int]],
        centers:list[list[float]],
        parallel: bool = False,
        n_cores: int = -1,
        blockwargs: dict = {},
        tol = 1e-7,
        max_iter: int = 200,
        method: str = "cg",
        seed: int | None = None)->xr.DataArray:
        """Finds the proper unitary matrix U_mnk to compute the MLWFs at each point in parameter space.

        Args:
            n_wannier (int, tuple[int, int]): If an int, the bands from n = 0 to n = n_wannier are used to generate n_wannier functions. 
            If a tuple, the bands from n = n_wannier[0] to n_wannier[1] are used.
            centers (list[list[float]]): The center of each WF, one [x, ...] per function, e.g. [[x0, y0], [x1, y1]].
            parallel (bool, optional): Wheter to parallelize the whole function. Defaults to False.
            n_cores (int, optional): Numbers of cores to use in case of parallelization. Defaults to -1.
            blockwargs (dict, optional): Arguments to pass on to the Bloch-Schrödinger solver constructor function. Defaults to {}.
            tol (_type_, optional): Convergence threshold on the relative decrease of the spread. Defaults to 1e-7.
            max_iter (int, optional): Hard cap on the minimization iterations per parameter point. Defaults to 200.
            method (str, optional): 'cg' for conjugate gradient, 'sd' for plain steepest descent. Defaults to 'cg'.
            seed (int, optional): Seed for the random spreads of the initial gaussians. The minimization
            starts from a randomized guess, so without a seed two identical calls land on slightly
            different matrices; pass one to make a computation reproducible. Defaults to None.

        Returns:
            xr.DataArray: The unitary matrix U_mnk with additional parameter dimensions.
        """
        print("Computing the Bloch functions...")
        
        if isinstance(n_wannier, int):
            self.n_wannier = [0,n_wannier]
            self.compute_bloch(**blockwargs)
        else:
            self.n_wannier = n_wannier
            self.compute_bloch(**blockwargs)
        
        self.nbands = self.n_wannier[1]-self.n_wannier[0]

        paramcoords = {
            dim:self.eigve.coords[dim] for dim in self.eigve.dims
            if dim not in ['m', *self.kb_dims, *self.spatial_dims]
        }
        
        allcoords = {
            **paramcoords,
            "m":self.eigve.m,
            "n":self.eigve.m.rename('n').rename({'m':'n'}),
            **{d: self.eigve.coords[d] for d in self.kb_dims},
        }
        
        shape = tuple(
            [coord.shape[0] for coord in allcoords.values()]
        )
        
        U_tot_mnk = xr.DataArray(
            np.zeros(shape, dtype=complex),
            coords = allcoords
        )
        
        # Flattening parameter space
        indexes = [np.arange(coord.shape[0]) for coord in paramcoords.values()]
        indexGrid = np.meshgrid(*indexes, indexing="ij")
        indexGrid = [grid.reshape(-1) for grid in indexGrid]
        selections = [
            {
                dim:paramcoords[dim][tup[i]].item() for i, dim in enumerate(paramcoords)
            } 
            for tup in zip(*indexGrid)
        ]
        
        if len(selections) == 0:
            selections = [{}]
        
        n_tot = len(selections)
                
        # One generator per parameter point, all derived from the one seed, so that a run is
        # reproducible whether or not it ends up being parallelised
        seeds = np.random.SeedSequence(seed).spawn(n_tot)
        rngs = [np.random.default_rng(s) for s in seeds] if seed is not None else [None] * n_tot

        def f(x, rng):
            return self.compute_U_mnk(
                x, centers, tol, max_iter=max_iter, method=method, rng=rng, return_info=True
            )
        
        print(f"Computing {n_tot} sets of Wannier functions")
        if parallel:
            parallel = Parallel(n_jobs=min(n_cores, n_tot), return_as="list", verbose = 5)
            results = parallel(delayed(f)(x, r) for x, r in zip(selections, rngs))
        else:
            results = []
            with tqdm(total=n_tot) as pbar:
                for x, r in zip(selections, rngs):
                    results += [f(x, r)]
                    pbar.update(1)

        infos = [info for _, info in results]
        for i in range(n_tot):
            U_tot_mnk.loc[selections[i]] = results[i][0]

        n_failed = sum(not info["converged"] for info in infos)
        iterations = [info["iterations"] for info in infos]
        print(
            f"Minimization: {sum(iterations) / n_tot:.0f} iterations on average, "
            f"spread between {min(i['spread'] for i in infos):.4g} and "
            f"{max(i['spread'] for i in infos):.4g}"
            + (f", {n_failed} point(s) hit max_iter" if n_failed else "")
        )

        return U_tot_mnk

    def compute_wannier(
        self,
        U_mnk:xr.DataArray,
        bounds: list[tuple[int, int]],
        coarsen: tuple[int] | None = None,
        )->tuple[Potential, xr.DataArray]:
        """Compute the WFs profiles from a given unitary matrix.

        Args:
            U_mnk (xr.DataArray): The unitary matrix to use, its dimensions must match those of the corresponding Bloch vectors.
            bounds (list[tuple[int, int]]): One pair per axis, giving the range of unit cells along that
            lattice vector over which to extend the WFs computation.
            coarsen (tuple[int], optional): Wheter to coarsen the resolution of the mode profile, one factor
            per axis. The factors must be dividers of the potential's resolution. See xarray coarsen function
            for more infos. Defaults to no coarsening.

        Returns:
            tuple[Potential, xr.DataArray]: The extended potential for plotting/Hamiltonian computation as well as the MLWFs profiles.

        Raises:
            ValueError: If bounds or coarsen do not match the potential's dimensionality.
        """
        if coarsen is None:
            coarsen = tuple([1] * self.n_dims)
        if len(bounds) != self.n_dims or len(coarsen) != self.n_dims:
            raise ValueError(
                f"For a {self.n_dims}D potential, bounds and coarsen must both have {self.n_dims} "
                f"entries, got {len(bounds)} and {len(coarsen)}"
            )

        if any(c != 1 for c in coarsen):
            coarse_eig = self.eigve.coarsen(
                {dim: c for dim, c in zip(self.spatial_dims, coarsen)},
                coord_func='min'
            ).mean()
        else:
            coarse_eig = self.eigve

        n_coarse = [
            self.potential.resolution[i] // coarsen[i] for i in range(self.n_dims)
        ]
        n_cells = [bounds[i][1] - bounds[i][0] for i in range(self.n_dims)]
        n_tot = [n_coarse[i] * n_cells[i] for i in range(self.n_dims)]

        coords = {
            dim: coarse_eig.coords[dim]
            for dim in coarse_eig.dims
            if dim not in [*self.spatial_dims, *self.kb_dims]
        }

        coords.update(
            {
                dim: np.linspace(
                    bounds[i][0] - 1 / 2, bounds[i][1] - 1 / 2, n_tot[i], endpoint=False
                )
                + 1 / n_coarse[i] / 2
                for i, dim in enumerate(self.spatial_dims)
            }
        )

        shape = tuple(
            [coord.shape[0] for coord in coords.values()]
        )

        wannier = xr.DataArray(
            np.zeros(shape, dtype = complex),
            coords=coords
        )

        # Cartesian coordinates over the whole tiled region, r_c = sum_i a[i][c] * a_i
        wannier = wannier.assign_coords(
            {
                name: sum(
                    self.potential.a[i][c] * wannier.coords[dim]
                    for i, dim in enumerate(self.spatial_dims)
                )
                for c, name in enumerate(self.coord_names)
            }
        )

        print("Computing the mode profiles...")
        k_grid = [kc.transpose(*self.kb_dims) for kc in self.k]
        k_indexes = list(itertools.product(*[range(n) for n in self.nb]))

        # Lay the spatial axes out last and contiguous, so a periodic block can simply be tiled over them
        lead_dims = [d for d in wannier.dims if d not in self.spatial_dims]
        wannier = wannier.transpose(*lead_dims, *self.spatial_dims)
        acc = np.zeros(wannier.shape, dtype=complex)

        # exp(-i k.r) is evaluated over the whole tiled region, so the cells need no separate handling
        r_tiled = [
            wannier.coords[name].transpose(*self.spatial_dims).values
            for name in self.coord_names
        ]
        reps = [1] * len(lead_dims) + n_cells

        for ik in tqdm(k_indexes):
            lcK = {dim: ik[i] for i, dim in enumerate(self.kb_dims)}
            k_cart = [float(kc[ik]) for kc in k_grid]

            # The periodic part of the Bloch function is identical in every unit cell, so it is built
            # once on the base cell and repeated, rather than re-derived for each cell in turn.
            psi = (U_mnk[lcK] * coarse_eig[lcK]).sum("m").rename({"n": "m"})
            psi = psi.transpose(*lead_dims, *self.spatial_dims).values

            phase = np.exp(
                -1j * sum(k_cart[c] * r_tiled[c] for c in range(self.n_dims))
            )
            acc += np.tile(psi, reps) * phase

        wannier = wannier.copy(data=acc)

        tiled_pot = self.potential.coarsen(coarsen).tile(bounds)

        wannier = wannier.rename({"m":"n"})
        wannier = wannier / ((abs(wannier) ** 2).sum(self.spatial_dims) * tiled_pot.get_dS()) ** 0.5

        return tiled_pot, wannier
        
  
