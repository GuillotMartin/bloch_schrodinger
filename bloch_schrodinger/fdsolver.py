import itertools

import numpy as np
import scipy.sparse as sps
import xarray as xr
from scipy.sparse.linalg import eigsh

from bloch_schrodinger.potential import Potential
from bloch_schrodinger.progress import bar, parallel_map
from bloch_schrodinger.utils import empty_from_coords


def real(arr: xr.DataArray) -> xr.DataArray:
    return xr.apply_ufunc(np.real, arr)


def imag(arr: xr.DataArray) -> xr.DataArray:
    return xr.apply_ufunc(np.imag, arr)


def check_name(name: str, n_dims: int):
    """Check whether the name is a valid one, and raises an error if not.

    Args:
        name (str): The name to check
        n_dims (int): The dimensionality of the solver, used to generate the list of forbidden names.

    Raises:
        ValueError: If the name is forbidden
    """

    coord_names = ["x", "y", "z"][:n_dims]
    # Names that would otherwise collide with dims/coords used internally
    forbidden_names = (
        ["field", "band", "lap"]
        + [f"a{i + 1}" for i in range(n_dims)]
        + coord_names
        + [f"d{c}" for c in coord_names]
    )
    if name in forbidden_names:
        raise ValueError(
            f"{name} is not a valid name for the object, as it is already used. The forbidden names are: {forbidden_names}"
        )


class FDSolver:
    """The FDSolver class handles the heavy lifting of this package, constructing the Hamiltonian matrix and performing its efficient diagonalization, using a finite difference scheme"""

    def __init__(
        self,
        potentials: Potential | list[Potential],
        alphas: float | xr.DataArray | list[float | xr.DataArray],
    ):
        """Instantiate the solver.

        Args:
            potentials (Union[Potential,list[Potential]]): The potential felt by each field. They must all be defined on the same grid.
            A single potential can also be passed for a scalar equation.
            alphas (Union[Union[float, xr.DataArray], list[Union[float, xr.DataArray]]]): The kinetic energy coefficient hbar²/2m for each field.
            A single coefficient can be passed for a scalar equation.

        Raises:
            ValueError: If potentials and alphas don't have the same length, or if the potentials don't
            all share the same dimensionality and resolution.
        """

        if isinstance(potentials, Potential) and (
            isinstance(alphas, (float, int, xr.DataArray))
        ):
            self.potentials = [potentials]
            self.alphas = [alphas]
            self.nb = 1
        else:
            self.potentials = potentials
            self.alphas = alphas
            if len(potentials) == len(alphas):
                self.nb = len(potentials)
            else:
                raise ValueError("potentials and alphas not of the same length")

        self.n_dims = self.potentials[0].n_dims
        self.spatial_dims = [f"a{i + 1}" for i in range(self.n_dims)]
        if any(
            pot.n_dims != self.n_dims or pot.resolution != self.potentials[0].resolution
            for pot in self.potentials
        ):
            raise ValueError(
                "All potentials must share the same dimensionality and resolution"
            )

        # storing all parameter coordinates from potentials and alphas. The final solver will run on all these dimensions.
        self.allcoords = {}
        for pot in self.potentials:
            coords_pot = {
                dim: ["potential", pot.V.coords[dim]]
                for dim in pot.V.dims
                if dim not in self.spatial_dims
            }
            self.allcoords.update(coords_pot)

        for alph in self.alphas:
            if isinstance(alph, xr.DataArray):
                for dim in alph.dims:
                    check_name(dim, self.n_dims)
                    coords_alpha = {
                        dim: ["alpha", alph.coords[dim]] for dim in alph.dims
                    }
                    self.allcoords.update(coords_alpha)

        self.k = (
            [0] * self.n_dims
        )  # The solver assumes k = 0 along every axis if not specified otherwise

        self.a_coords = [
            self.potentials[0].V.coords[f"a{i + 1}"] for i in range(self.n_dims)
        ]
        self.a = self.potentials[0].a  # The lattice vectors
        self.n_a = [
            self.potentials[0].V.sizes[f"a{i + 1}"] for i in range(self.n_dims)
        ]  # discretization along each axis
        self.np = int(np.prod(self.n_a))  # Number of mesh sampling points

        self.n = (
            self.np * self.nb
        )  # Total number of points, when counting all the fields

        self.g = np.linalg.inv(
            self.a @ self.a.T
        )  # The metric, important to compute derivative operators

        self.da = self.potentials[0].da  # length increments along each axis

        # Initializing the coupling evalution context and the coupling expressions
        self.coupling_context = {"np": np, "real": real, "imag": imag}
        self.couplings = []

        self.maxsearch = min(
            min(self.n_a) // 2 - 2, 10
        )  # The maximal distance (inf norm) to look for shells.
        self.create_stencil()
        # computing the sparse matrices needed
        self.potential_matrix()
        self.compute_grad()
        self.compute_laplacian()

    def __repr__(self) -> str:
        shape = {dim: len(self.allcoords[dim][1].data) for dim in self.allcoords}
        return f"Solver object \n size ({self.n}, {self.n}), with {self.nb} field(s) \n dimensions: {shape}"

    def potential_matrix(self):
        """Prepare the potentials in a shape that can be instantiated as sparse matrices very efficiently"""

        # flattening the potentials
        Vflats = [pot.V.stack(idiag=self.spatial_dims) for pot in self.potentials]

        # Expanding each potential dimensions to the full parameter space, composed of all dimensions of each potential
        for i, V in enumerate(Vflats):
            V = V.drop_vars(["idiag", *self.spatial_dims]).assign_coords(
                idiag=("idiag", np.arange(i * self.np, (i + 1) * self.np))
            )
            for d, coords in self.allcoords.items():
                if coords[0] == "potential":
                    if d not in V.dims:
                        V = V.expand_dims({d: coords[1]})
            Vflats[i] = V

        # data in the shape (...,self.n) with the first dimensions being all the parameter dimensions of potentials
        self.potential_data = xr.concat(Vflats, dim="idiag")

    def create_stencil(self):
        """Creates the stencil for the gradient and laplacian operator definitions. For 2D, follows
        https://arxiv.org/pdf/0708.0650 sec. 3.2 exactly (2nd-moment isotropy plus a 4th-order anisotropy
        cancellation). For other dimensionalities, only the leading-order (2nd-moment) isotropy condition
        sum_b w_b b_i b_j = delta_ij is enforced, which is dimension-agnostic but less refined than the 2D case."""
        coords = self.potentials[0].coords  # cartesian coordinates, one array per axis
        center_idx = tuple(n // 2 for n in self.n_a)
        origin = np.array([coords[d][center_idx].item() for d in range(self.n_dims)])

        lim = self.maxsearch
        # Computing the displacements in index and cartesian coordinates for a few shells
        offsets, disp, Dist = [], [], []
        for offset in itertools.product(range(-lim, lim + 1), repeat=self.n_dims):
            if any(offset):
                idx = tuple(center_idx[d] + offset[d] for d in range(self.n_dims))
                point = np.array([coords[d][idx].item() for d in range(self.n_dims)])
                vec = point - origin
                offsets += [offset]
                disp += [vec]
                Dist += [float(np.linalg.norm(vec))]

        offsets = np.array(offsets)
        disp = np.array(disp)
        Dist = np.array(Dist)

        # Sorting all the points by ascending distance to the central one
        sorting = np.argsort(Dist)
        offsets_sorted = offsets[sorting]
        disp_sorted = disp[sorting]
        Dist_sorted = Dist[sorting]

        # Sorting the points by shell number
        shell_distance, start_shell, shell_index, n_in_shell = np.unique(
            Dist_sorted.round(decimals=8),
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )

        def shell_moment(s, i, j):
            m = slice(start_shell[s], start_shell[s] + n_in_shell[s])
            return np.sum(disp_sorted[m, i] * disp_sorted[m, j])

        def shell_moment4(s):
            # Extra 2D-only 4th-order anisotropy cancellation
            m = slice(start_shell[s], start_shell[s] + n_in_shell[s])
            x, y = disp_sorted[m, 0], disp_sorted[m, 1]
            return np.sum(x**4 - y**4), np.sum(x**2 * y**2 - (1 / 3) * x**4)

        # The moment conditions defining an isotropic Laplacian: sum_b w_b b_i b_j = delta_ij
        pairs = [(i, j) for i in range(self.n_dims) for j in range(i, self.n_dims)]
        q = [1.0 if i == j else 0.0 for i, j in pairs]
        if self.n_dims == 2:
            q += [0.0, 0.0]
        q = np.array(q)

        # Solve for the weights, keep adding shells until solvable
        solved = False
        nshell = 1
        while not solved:
            A_js = [[shell_moment(s, i, j) for s in range(nshell)] for i, j in pairs]
            if self.n_dims == 2:
                fourth = [shell_moment4(s) for s in range(nshell)]
                A_js += [[f[0] for f in fourth], [f[1] for f in fourth]]
            A_js = np.array(A_js)

            # Minimum-norm least-squares solve; robust to A_js being rank-deficient
            # at some shell counts (common in 3D, where symmetric shells can have
            # exactly zero cross-axis moments).
            w = np.linalg.pinv(A_js) @ q
            if np.all(np.isclose(A_js @ w, q)):
                solved = True
                self.n_shell = nshell
            else:
                nshell += 1

        weights = []
        neighbors = []
        neighbors_indexes = []
        for s in range(nshell):
            for m in range(n_in_shell[s]):
                indx = start_shell[s] + m
                neighbors_indexes += [tuple(offsets_sorted[indx])]
                neighbors += [list(disp_sorted[indx])]
                weights += [w[s]]

        self.stencil_size = len(weights)
        self.weights = xr.DataArray(weights, coords={"b": np.arange(self.stencil_size)})
        self.neighbors = xr.DataArray(
            neighbors,
            coords={"b": np.arange(self.stencil_size), "axis": np.arange(self.n_dims)},
        )
        self.neighbors_indexes = xr.DataArray(
            neighbors_indexes,
            coords={"b": np.arange(self.stencil_size), "axis": np.arange(self.n_dims)},
        )

        # --- Find the second order weights ---
        def compute_hessian_weights(b_vectors):
            """
            Compute finite-difference weights for the Hessian (second derivatives)
            given a list of displacement vectors b_vectors.

            Parameters
            ----------
            b_vectors : ndarray of shape (N, d)
                List of N displacement vectors in d dimensions.

            Returns
            -------
            weights : dict
                Dictionary with keys (i,j) for each second derivative component,
                values are arrays of length N giving the weights omega_{b,ij}.
            """
            b_vectors = np.array(b_vectors)
            N, d = b_vectors.shape
            weights = {}

            # Generate all second derivative index pairs (i <= j for symmetry)
            index_pairs = [(i, j) for i in range(d) for j in range(i, d)]

            # Precompute the b_m * b_n for each b vector
            # shape: (N, d, d)
            b_outer = np.einsum("bi,bj->bij", b_vectors, b_vectors)

            for i, j in index_pairs:
                # Right-hand side: 2 * delta_{im} delta_{jn} from Taylor expansion
                # We want only one component to be 2, rest 0 in least-squares sense
                B = np.zeros((d, d))
                B[i, j] = 2.0

                # Flatten b_outer to shape (N, d*d) for least squares
                A = b_outer.reshape(N, d * d)

                # Flatten B to shape (d*d,)
                B_flat = B.flatten()

                # Solve least-squares: A^T w = B_flat
                w, residuals, rank, s = np.linalg.lstsq(A.T, B_flat, rcond=None)
                weights[(i, j)] = w
                # If i != j, enforce symmetry: omega_{ij} = omega_{ji}
                if i != j:
                    weights[(j, i)] = w

            return weights

        self.weights2 = compute_hessian_weights(neighbors)

    def compute_hopping(self, delta: tuple[int]) -> sps.spmatrix:
        """Compute the hopping matrix connecting each site to the site shifted by delta,
        taking care of the periodic boundaries.

        Args:
            delta (tuple[int]): The shift to apply along each axis (a1, a2, ...).

        Returns:
            sps.spmatrix: A (np,np) sparse matrix containing the hopping matrix.
        """
        indx_start = np.arange(self.np)
        idx_start = np.unravel_index(indx_start, self.n_a)

        idx_end = tuple(
            (idx_start[d] + delta[d]) % self.n_a[d] for d in range(self.n_dims)
        )

        indx_end = np.ravel_multi_index(idx_end, self.n_a)

        all_offset = indx_end - indx_start
        offsets = np.unique(all_offset)
        frac_diags = []
        for offset in offsets:
            frac_diags += [np.roll(np.where(all_offset == offset, 1, 0), offset)]

        spsdiag = sps.dia_matrix((frac_diags, offsets), shape=(self.np, self.np))
        return spsdiag

    def compute_grad(self):
        """Compute the gradient and derivation operators, one per axis (self.d[axis])"""
        ## We use the stencil formula for symmetric shells
        self.d = [
            sps.dia_matrix((self.np, self.np), dtype=float) for _ in range(self.n_dims)
        ]
        for i in range(self.stencil_size):
            delta = tuple(
                int(self.neighbors_indexes[i, d].item()) for d in range(self.n_dims)
            )
            hop = self.compute_hopping(delta) - sps.eye(self.np, self.np)
            w = self.weights[i].item()
            for d in range(self.n_dims):
                self.d[d] += hop * self.neighbors[i, d].item() * w

        # Named aliases for the common low-dimensional cases
        if self.n_dims >= 1:
            self.dx = self.d[0]
        if self.n_dims >= 2:
            self.dy = self.d[1]
        if self.n_dims >= 3:
            self.dz = self.d[2]

    def compute_laplacian(self):
        """Compute the laplacian and second order derivation operators, self.d2[(i,j)] for every axis pair"""

        self.lap = sps.dia_matrix((self.np, self.np), dtype=float)
        upper = [(i, j) for i in range(self.n_dims) for j in range(i, self.n_dims)]
        self.d2 = {
            pair: sps.dia_matrix((self.np, self.np), dtype=float) for pair in upper
        }

        for i in range(self.stencil_size):
            delta = tuple(
                int(self.neighbors_indexes[i, d].item()) for d in range(self.n_dims)
            )
            hop = self.compute_hopping(delta) - sps.eye(self.np, self.np)

            w = self.weights[i].item()

            self.lap += hop * w * 2
            for a, b in upper:
                self.d2[(a, b)] += hop * self.weights2[(a, b)][i]

        # mirror the off-diagonal entries so self.d2[(j,i)] is also directly accessible
        for a, b in upper:
            if a != b:
                self.d2[(b, a)] = self.d2[(a, b)]

        # Named aliases for the common 2D case
        if self.n_dims == 2:
            self.dx_sec = self.d2[(0, 0)]
            self.dy_sec = self.d2[(1, 1)]
            self.dxdy = self.d2[(0, 1)]
            self.dydx = self.d2[(1, 0)]

    def compute_full_operators(self, alphas: list[float]):
        """Compute all the total kinetic operators for each fields, as block sparse matrices

        Args:
            alphas (list[float]): The list of alphas to use
        """

        # Initialization as lists of blocks
        alphaIdent = [
            alphas[u] * sps.eye_array(n=self.np, m=self.np) for u in range(self.nb)
        ]
        alphaD = [
            [alphas[u] * self.d[axis] for u in range(self.nb)]
            for axis in range(self.n_dims)
        ]
        alphaLap = [alphas[u] * self.lap for u in range(self.nb)]
        # create the full matrices
        self.alphaIdent = sps.block_diag(alphaIdent)
        self.alphaD = [sps.block_diag(alphaD[axis]) for axis in range(self.n_dims)]
        self.alphaLap = sps.block_diag(alphaLap)

    def compute_kinetic(self, k: list[float]) -> sps.dia_matrix:
        """Compute the total kinetic operator.

        Args:
            k (list[float]): The k vector, one component per axis.
        """
        # Bloch substitution -grad^2 -> -(grad + ik)^2, expanded term by term
        kinetic = -self.alphaLap
        for axis in range(self.n_dims):
            kinetic = (
                kinetic
                + 2j * k[axis] * self.alphaD[axis]
                + k[axis] ** 2 * self.alphaIdent
            )
        return kinetic

    def set_reciprocal_space(self, k: list[float | xr.DataArray]):
        """Add the reciprocal space to the list of coordinates.

        Args:
            k (list[Union[float,xr.DataArray]]): The k vector components, one per axis. Each component can be
            a single float, a 1D xarray coordinate, or even a multidimensional coordinate.
        """
        for ki in k:
            if isinstance(ki, xr.DataArray):
                self.allcoords.update(
                    {dim: ["reciprocal", ki.coords[dim]] for dim in ki.dims}
                )

        self.k = list(k)

    def create_reciprocal_grid(self, k: list[float | np.ndarray] | None = None):
        """Create the k-space grid on which the eigenvalues and vectors will be computed

        Args:
            k (list[float or np.ndarray], optional): The values of k for the grid points, one per axis.
            Defaults to 0 along every axis.
        """
        if k is None:
            k = [0] * self.n_dims

        k_names = ["kx", "ky", "kz"][: self.n_dims]
        k_arrays = []
        for name, ki in zip(k_names, k):
            if isinstance(ki, float) or isinstance(ki, int):
                k_arrays += [
                    xr.DataArray(
                        np.array([ki]), coords={name: np.array([ki])}, dims=name
                    )
                ]
            else:
                k_arrays += [xr.DataArray(ki, coords={name: ki}, dims=name)]

        self.set_reciprocal_space(k_arrays)

    def add_coupling_parameter(
        self, name: str, parameter: complex | np.ndarray | xr.DataArray
    ):
        """Add a parameter to the coupling evaluation context or to the list of dimensions.

        Args:
            name (str): The name of the coupling parameter, if the parameter given is an xarray, its name will be overwritten by this argument. must be unique.
            parameter (Union[float,int,complex,xr.DataArray]): The value(s) of the parameter.
        """
        check_name(name, self.n_dims)

        if name in self.coupling_context:
            raise ValueError(f"{name} already present in the coupling context")

        if (
            isinstance(parameter, (float, complex, int))
        ):
            self.coupling_context.update({name: parameter})
        else:
            if isinstance(parameter, np.ndarray):
                parameter = xr.DataArray(
                    data=parameter, coords={name: parameter}, name=name
                )
            elif isinstance(parameter, xr.DataArray):
                parameter.rename(name)
            else:
                raise TypeError("Type of parameter not supported")

            self.allcoords.update({name: ["coupling", parameter]})

    def add_coupling_matrix(
        self, name: str, struct: sps.spmatrix, field1: int, field2: int
    ):
        """Add a sparse matrix to the coupling context, to be evaluated during Hamiltonian construction.
        The conjugate transpose term (field2, field1) is added automatically, so the resulting block is Hermitian.

        Args:
            name (str): The name of the matrix to call during evaluation, must be unique.
            struct (sps.spmatrix): A (self.np,self.np) sparse matrix containing the structure of the coupling term.
            field1 (int): The origin field of the coupling.
            field2 (int): The target field of the coupling.
        """

        check_name(name, self.n_dims)

        shape = [[None, None], [None, None]]
        shape[field1][field2] = struct
        shape[field2][field1] = struct.transpose().conj()

        matrix = sps.block_array(shape)
        self.coupling_context.update({name: matrix})

    def add_coupling(self, expr: str):
        """Add a coupling expression to the list of couplings. The coupling expression is a string that will be evaluated at Hamiltonian building phase.
        It can use all parameters and matrices set by 'add_coupling_parameter' and 'add_coupling_matrix', as well as the other parameters of the potentials and alphas.
        Be careful to not multiply a matrix by an imaginary number, as it will break hermiticity. You should instead create a purely imaginary Hermitian matrix then multiply it by a real value

        Args:
            expr (str): The string to evaluate.
        """
        self.couplings += [expr]

    def add_on_site_coupling(self, expr: str, field1: int, field2: int):
        """Add a simple, local coupling term between two field. The complex conjugate term is applied automatically.

        Args:
            expr (str): An expression describing the stength of the coupling, must only use parameters that are in (or will be added to) the context manager.
            field1 (int): The origin field of the coupling.
            field2 (int): The target field of the coupling.
        """

        matrix = sps.eye(self.np, self.np)
        mat_name = f"id_{field1}{field2}"
        imat_name = f"i_id_{field1}{field2}"
        self.add_coupling_matrix(mat_name, matrix, field1, field2)
        self.add_coupling_matrix(imat_name, 1j * matrix, field1, field2)

        self.add_coupling(f"real({expr}) * {mat_name} + imag({expr}) * {imat_name}")

    def add_TETM(self, expr: str, field1: int, field2: int):
        """Add a TETM splitting term between fields 1 and 2. TE/TM splitting is a 2D vectorial-wave concept,
        so this is only defined for 2D solvers.

        Args:
            expr (str): An expression describing the stength of the coupling, must only use parameters that are in (or will be added to) the context manager.
            field1 (int): The origin field of the coupling.
            field2 (int): The target field of the coupling.

        Raises:
            ValueError: If the solver is not 2D.
        """
        if self.n_dims != 2:
            raise ValueError(
                "add_TETM is only defined for 2D solvers, as TE/TM splitting has no general nD analog"
            )

        f1, f2 = field1, field2  # shorten names for expression length

        # adding the real and imaginary main diagonal
        matrix = sps.eye(self.np, self.np)
        mat_name = f"id_{f1}{f2}"
        self.add_coupling_matrix(mat_name, matrix, field1, field2)

        mat_name = f"i_id_{f1}{f2}"
        self.add_coupling_matrix(mat_name, 1j * matrix, field1, field2)

        self.add_coupling_matrix(f"dx_{f1}{f2}", self.dx, f1, f2)
        self.add_coupling_matrix(f"dy_{f1}{f2}", self.dy, f1, f2)

        self.add_coupling_matrix(f"dx_sec_{f1}{f2}", self.dx_sec, f1, f2)
        self.add_coupling_matrix(f"dy_sec_{f1}{f2}", self.dy_sec, f1, f2)

        # we have to precompute the proper matrix multiplications in order to keep hermiticity
        self.add_coupling_matrix(f"idx_{f1}{f2}", 1j * self.dx, f1, f2)
        self.add_coupling_matrix(f"idy_{f1}{f2}", 1j * self.dy, f1, f2)
        self.add_coupling_matrix(f"idxdy_{f1}{f2}", 1j * self.dxdy, f1, f2)
        self.add_coupling_matrix(f"idydx_{f1}{f2}", 1j * self.dydx, f1, f2)

        # directly add kx and ky equal to 0 to the manager, it will be overwritten if kx and ky are defined latter.
        self.coupling_context.update({"kx": 0, "ky": 0})

        # The expression of the TE/TM splitting operator
        expr_coup = (
            f"(ky**2 - kx**2)*id_{f1}{f2} + 2*kx*ky*i_id_{f1}{f2} + "  # zeroth-order term
            + f"2 * (kx * (idx_{f1}{f2} + dy_{f1}{f2}) + ky * (dx_{f1}{f2} - idy_{f1}{f2})) + "  # first order term
            + f"dx_sec_{f1}{f2} - dy_sec_{f1}{f2} - idxdy_{f1}{f2} - idydx_{f1}{f2}"  # second order term
        )

        self.add_coupling(f"{expr} * ({expr_coup})")

    def initialize_eigva(self, n_eigva: int) -> xr.DataArray:
        """Initialize the array containing the eigenvalues

        Args:
            n_eigva (int): The number of eigenvalues to compute

        Returns:
            xr.DataArray: An empty DataArray with the proper shape.
        """
        eigva_coords = [coord[1] for coord in self.allcoords.values()]

        eigva_coords += [
            xr.DataArray(np.arange(n_eigva), coords={"band": np.arange(n_eigva)})
        ]

        return empty_from_coords(eigva_coords, float, "eigva")

    def initialize_eigve(self, n_eigva: int, stack: bool = True) -> xr.DataArray:
        """Initialize the array containing the flattened eigenvectors

        Args:
            n_eigva (int): The number of eigenvalues to compute
            stack (bool): Whether to stack the eigenvectors along a single 'component' dimension. Some children classes don't want that. default to True
        Returns:
            xr.DataArray: An empty DataArray with the proper shape.
        """
        eigve_coords = [coord[1] for coord in self.allcoords.values()]

        eigve_coords += [
            xr.DataArray(np.arange(self.nb), coords={"field": np.arange(self.nb)})
        ]

        eigve_coords += self.a_coords

        eigve_coords += [
            xr.DataArray(np.arange(n_eigva), coords={"band": np.arange(n_eigva)})
        ]

        eigve = empty_from_coords(eigve_coords, complex, "eigve_flat")

        eigve = eigve.assign_coords(
            {
                self.potentials[0].coord_names[d]: self.potentials[0].coords[d]
                for d in range(self.n_dims)
            }
        )
        if stack:
            eigve = eigve.stack(component=("field", *self.spatial_dims)).transpose(
                ..., "component", "band"
            )
        return eigve

    def make_alpha_list(self, sel: dict) -> list[float]:
        """Select the proper alpha value for each field given a dimension selection

        Args:
            sel (dict): The selection of coordinate index for alphas.

        Raises:
            TypeError: Raises an error if one of the alpha given to the constructor is not of type int, float or DataArray.

        Returns:
            list[float]: A list of single valued alphas.
        """
        alphas = []
        for u in range(len(self.alphas)):
            if isinstance(self.alphas[u], float) or isinstance(self.alphas[u], int):
                alphas += [self.alphas[u]]
            elif isinstance(self.alphas[u], xr.DataArray):
                sub_sel = {
                    dim: value
                    for dim, value in sel.items()
                    if dim in self.alphas[u].dims
                }
                alphas += [float(self.alphas[u].sel(sub_sel).data)]
            else:
                raise TypeError(
                    f"{u}-th alpha term not of a recognized type (int, float or xr.DataArray)"
                )
        return alphas

    def create_hamiltonian(
        self,
        potential_sel: dict,
        alpha_sel: dict,
        reciprocal_sel: dict,
        coupling_sel: dict,
    ) -> sps.spmatrix:
        """Return a (N, N) sparse matrix containing the Hamiltonian for a given set of parameters

        Args:
            potential_sel (dict): The parameters selection for the potential.
            alpha_sel (dict): The parameters selection for the kinetic factor.
            reciprocal_sel (dict): The position in k-space.
            coupling_sel (dict): The parameters selection for the additional coupling terms.

        Returns:
            sps.spmatrix: The Hamiltonian.
        """

        # Selecting only one value for each potential dimension; the selection stays empty if there are no potential dimensions

        # The potential is a diagonal matrix, which we stored as a data array.
        potdiag = self.potential_data.sel(potential_sel).data
        potential_matrix = sps.diags(potdiag, offsets=0)

        # Same for the alphas, using the 'make_alpha_list' class method
        alphas = self.make_alpha_list(alpha_sel)
        self.compute_full_operators(alphas)

        # Resolve each k component: a fixed value, or selected from a swept coordinate
        k = []
        for ki in self.k:
            if isinstance(ki, xr.DataArray):
                ksel = {
                    key: value
                    for key, value in reciprocal_sel.items()
                    if key in ki.dims
                }
                k += [float(ki.sel(ksel).data)]
            else:
                k += [ki]
        kinetic_matrix = self.compute_kinetic(k)

        total_sel = {
            **potential_sel,
            **alpha_sel,
            **reciprocal_sel,
            **coupling_sel,
        }  # aggregating all the values of each selections
        ham = (
            kinetic_matrix + potential_matrix
        )  # The initial Hamiltonian contains only the kinetic operator and the potential operator

        # --- Add the coupling terms to the Hamiltonian ---
        # Overlay total_sel on a copy, so concurrent calls don't mutate shared solver state
        eval_context = {**self.coupling_context, **total_sel}
        for coupling in self.couplings:
            ham += eval(coupling, {"__builtins__": {}}, eval_context)

        return ham

    def normalize(
        self, eigve: xr.DataArray | np.ndarray, norm: float = 1
    ) -> xr.DataArray:
        """Normalize the eigenvector array to a specified value in real-space units.

        Args:
            eigve (xr.DataArray, np.ndarray): The eigenvector array
            norm (float, optional): The norm of the array. Defaults to 1.

        Returns:
            xr.DataArray, np.ndarray
        """
        if isinstance(eigve, xr.DataArray):
            if "field" in eigve.dims:
                dims = self.spatial_dims + ["field"]
            elif "component" in eigve.dims:
                dims = ["component"]
            else:
                dims = self.spatial_dims

            normed = eigve / (abs(eigve) ** 2).sum(dims) ** 0.5

        else:
            normed = eigve / (np.abs(eigve) ** 2).sum()

        return normed * (norm / self.potentials[0].get_dS()) ** 0.5

    def solve(
        self,
        n_eigva: int,
        keep_vecs: bool = True,
        parallel: bool = False,
        n_cores: int = -1,
        verbose: bool = True,
        **kwargs,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Solve the eigenvalue problem at every points of the parameter space, using the scipy.sparse.eigsh function.

        Args:
            n_eigva (int): The number of eigenvalues to compute.
            keep_vecs (bool, optional): Whether to also compute and return the eigenvectors. Default to True.
            parallel (bool, optional): Whether to parallelize the solver with joblib. Default to False.
            n_cores (int, optional): The number of cores to use for the solver, set to -1 to use all cores available. default to -1.
            verbose (bool, optional): Whether to plot a progress bar over the diagonalizations. Default to True.
        Returns:
            tuple[xr.DataArray, xr.DataArray] or xr.DataArray: the eigenvalues, and the eigenvectors if keep_vecs is True
            (otherwise the eigenvalues alone).

        Additional kwargs:
            phase0 (tuple[float,...,int]): The position at which to fix the phase of each eigenvector, in a1, a2, ..., field basis.
            Defaults to (0, ..., 0) (the origin, field 0).
        """

        # Create empty DataArrays to store the eigenvalues and vectors
        eigva = self.initialize_eigva(n_eigva)
        if keep_vecs:
            eigve = self.initialize_eigve(n_eigva)

        # Create lists of dimensions. The dimensions are separated by type, for efficient Hamiltonian matrix construction.
        potential_dims = [
            dim for dim in self.allcoords if self.allcoords[dim][0] == "potential"
        ]
        reciprocal_dims = [
            dim for dim in self.allcoords if self.allcoords[dim][0] == "reciprocal"
        ]
        alphas_dims = [
            dim for dim in self.allcoords if self.allcoords[dim][0] == "alpha"
        ]
        couplings_dims = [
            dim for dim in self.allcoords if self.allcoords[dim][0] == "coupling"
        ]

        # Intializing indexes to loop over.
        potential_index, alpha_index, reciprocal_index, coupling_index = (
            [()],
            [()],
            [(0, 0)],
            [()],
        )

        # Stacking the dimensions of each type and replacing the associated index. If there is no dimensions of a given type,
        # the corresponding index is kept as the default defined above
        if len(potential_dims) > 0:
            eigva = eigva.stack(potdims=potential_dims)
            potential_index = eigva.potdims.to_index()
            if keep_vecs:
                eigve = eigve.stack(potdims=potential_dims)
        if len(alphas_dims) > 0:
            eigva = eigva.stack(alphadims=alphas_dims)
            alpha_index = eigva.alphadims.to_index()
            if keep_vecs:
                eigve = eigve.stack(alphadims=alphas_dims)
        if len(reciprocal_dims) > 0:
            eigva = eigva.stack(recdims=reciprocal_dims)
            reciprocal_index = eigva.recdims.to_index()
            if keep_vecs:
                eigve = eigve.stack(recdims=reciprocal_dims)
        if len(couplings_dims) > 0:
            eigva = eigva.stack(coupdims=couplings_dims)
            coupling_index = eigva.coupdims.to_index()
            if keep_vecs:
                eigve = eigve.stack(coupdims=couplings_dims)

        # The total number of matrix to diagonalize, used for the progress bar
        n_tot = len(eigva.sel(band=0).data.reshape(-1))

        # Initializing the progress bar
        potential_sels, reciprocal_sels, coupling_sels, alpha_sels = [], [], [], []

        # Looping over everything
        for pots in potential_index:
            for alphs in alpha_index:
                for recs in reciprocal_index:
                    for coups in coupling_index:
                        # Selecting only one value for each potential dimension; the selection stays empty if there are no potential dimensions
                        potential_sels += [
                            (
                                dict(zip(potential_index.names, pots))
                                if hasattr(potential_index, "names")
                                else {}
                            )
                        ]
                        alpha_sels += [
                            (
                                dict(zip(alpha_index.names, alphs))
                                if hasattr(alpha_index, "names")
                                else {}
                            )
                        ]
                        reciprocal_sels += [
                            (
                                dict(zip(reciprocal_index.names, recs))
                                if hasattr(reciprocal_index, "names")
                                else {}
                            )
                        ]
                        coupling_sels += [
                            (
                                dict(zip(coupling_index.names, coups))
                                if hasattr(coupling_index, "names")
                                else {}
                            )
                        ]

        sels = [
            {**p, **r, **c, **a}
            for p, r, c, a in zip(
                potential_sels, reciprocal_sels, coupling_sels, alpha_sels
            )
        ]

        e, X = eigsh(
            self.create_hamiltonian(
                potential_sels[0], alpha_sels[0], reciprocal_sels[0], coupling_sels[0]
            ),
            k=n_eigva,
            which="SA",
        )

        def x(pot_sel, alpha_sel, rec_sel, cou_sel):
            ham = self.create_hamiltonian(pot_sel, alpha_sel, rec_sel, cou_sel)
            return eigsh(ham, k=n_eigva, v0=X[:, 0], which="SA")

        args = list(
            zip(potential_sels, alpha_sels, reciprocal_sels, coupling_sels)
        )

        if parallel:
            results = parallel_map(
                x,
                args,
                n_jobs=min(n_cores, n_tot),
                desc="Diagonalizing",
                unit="matrix",
                verbose=verbose,
            )
        else:
            results = [
                x(p, a, r, c)
                for p, a, r, c in bar(
                    args, desc="Diagonalizing", unit="matrix", verbose=verbose
                )
            ]

        for i in range(n_tot):
            eigvals, eigvecs = results[i][0], results[i][1]

            idx = eigvals.argsort()
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]

            eigva.loc[sels[i]] = eigvals
            if keep_vecs:
                eigve.loc[sels[i]] = eigvecs

        if len(potential_dims) > 0:
            eigva = eigva.unstack(dim="potdims")
            if keep_vecs:
                eigve = eigve.unstack(dim="potdims")
        if len(alphas_dims) > 0:
            eigva = eigva.unstack(dim="alphadims")
            if keep_vecs:
                eigve = eigve.unstack(dim="alphadims")
        if len(reciprocal_dims) > 0:
            eigva = eigva.unstack(dim="recdims")
            if keep_vecs:
                eigve = eigve.unstack(dim="recdims")
        if len(couplings_dims) > 0:
            eigva = eigva.unstack(dim="coupdims")
            if keep_vecs:
                eigve = eigve.unstack(dim="coupdims")

        if keep_vecs:
            eigve = eigve.unstack(dim="component").rename("eigve")

            pos0 = kwargs.get("phase0", (0.01,) * self.n_dims + (0,))
            sel0 = {self.spatial_dims[d]: pos0[d] for d in range(self.n_dims)}
            sel0["field"] = pos0[self.n_dims]

            eigve = eigve * xr.ufuncs.exp(
                -1j * xr.ufuncs.angle(eigve.sel(sel0, method="nearest"))
            )
            eigve = eigve.assign_coords(
                {
                    self.potentials[0].coord_names[d]: self.potentials[0].coords[d]
                    for d in range(self.n_dims)
                }
            )

        if keep_vecs:
            return eigva.squeeze(), self.normalize(eigve).squeeze()
        else:
            return eigva.squeeze()


if __name__ == "__main__":
    import time as time

    import matplotlib.pyplot as plt
    from plotting import plot_eigenvector

    res = (200, 200)

    a = 2.5
    a1 = np.array([-(3**0.5) / 2 * a, 3 / 2 * a])  # 1st lattice vector
    a2 = np.array([3**0.5 / 2 * a, 3 / 2 * a])

    # P = Potential([[3, -20], [3, 20]], resolution=res, v0=0)
    P = Potential([[10, 0], [0, 10]], resolution=res, v0=0)
    # P = Potential([a1,a2], resolution=res, v0=0)
    P.V = P.V.x**2 + P.V.y**2

    solv = FDSolver(P, 1)

    eigva, eigve = solv.solve(1)

    plot_eigenvector([[abs(eigve) ** 2]], [[P]], [["amplitude"]])
    plt.show()
    

