import warnings

import numpy as np
import xarray as xr
from numpy.linalg import inv
from scipy.fft import fftn, fftshift, ifftshift
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh

from bloch_schrodinger.potential import Potential
from bloch_schrodinger.progress import bar, parallel_map
from bloch_schrodinger.utils import empty_from_coords


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
        ["field", "band", "g", "gp", "ij"]
        + [f"a{i + 1}" for i in range(n_dims)]
        + [f"pwka{i + 1}" for i in range(n_dims)]
        + coord_names
        + [f"pwk{c}" for c in coord_names]
        + [f"k{c}" for c in coord_names]
    )
    if name in forbidden_names:
        raise ValueError(
            f"{name} is not a valid name for the object, as it is already used. The forbidden names are: {forbidden_names}"
        )


class PWSolver:
    """The PWSolver class is the second main solver of the package, it solves the Bloch-schrödinger equation by plane wave expansion.
    It only supports scalar equations for now, as well as a constant kinetic term."""

    def __init__(
        self,
        potential: Potential,
        alpha: float,
        E_lim: float = None,
        dense_limit: int = 1000,
    ):
        """Instantiate the solver.

        Args:
            potential (Potential): The potential field.
            alpha (float): The kinetic energy coefficient hbar²/2m.
            Elim (float): The energy cut-off.
            dense_limit (int): The basis size above which the solver switches from a dense
            factorization to ARPACK. See 'diagonalize'. Defaults to 1000.
        """

        self.potential = potential
        self.dense_limit = dense_limit
        self.n_dims = potential.n_dims
        self.spatial_dims = [f"a{i + 1}" for i in range(self.n_dims)]
        self.coord_names = ["x", "y", "z"][: self.n_dims]
        self.pw_dims = [f"pwka{i + 1}" for i in range(self.n_dims)]
        self.pwk_names = [f"pwk{c}" for c in self.coord_names]

        self.potential.V = self.potential.V.transpose(..., *self.spatial_dims)
        self.alpha = alpha

        # storing all parameter coordinates from the potential. The final solver will run on all these dimensions.
        self.allcoords = {}
        coords_pot = {}
        for dim in potential.V.dims:
            if dim not in self.spatial_dims:
                check_name(dim, self.n_dims)
                coords_pot[dim] = ["potential", potential.V.coords[dim]]
        self.allcoords.update(coords_pot)

        # The solver assumes k = 0 along every axis if not specified otherwise
        self.k = [0] * self.n_dims

        self.a_coords = [potential.V.coords[d] for d in self.spatial_dims]
        self.a = potential.a  # The lattice vectors, one per row
        self.n_a = [
            potential.V.sizes[d] for d in self.spatial_dims
        ]  # discretization along each axis
        self.np = int(np.prod(self.n_a))  # Number of mesh sampling points

        self.da = potential.da  # length increments along each axis

        self.compute_b()
        self.compute_fV()

        self.E_lim = E_lim if E_lim is not None else (potential.V.max()-potential.V.min()).item()*2
        self.set_cutoff(self.E_lim)
        if E_lim is None:
            print(f"Energy cut-off not specified and automatically set to {self.E_lim:.3f}, expanding on {self.nGs} vectors")


    def __repr__(self) -> str:
        return f"Plane wave Solver object ({self.n_dims}D) \n E_lim = {self.E_lim:.3f} \n Expanded on {self.nGs} plane waves"

    def compute_b(self):
        """Compute the reciprocal vectors of the unit cell"""
        # Row i is b_i, the vector satisfying b_i . a_j = 2.pi.delta_ij
        self.b = 2 * np.pi * inv(self.a).T

    def compute_fV(self):
        """Compute the Fourier transform of the potential matrix"""

        # Plane wave index along each axis, centered on zero the same way fftshift centers the transform
        self.pwka = [
            xr.DataArray(idx, coords={name: idx})
            for name, idx in (
                (name, np.arange(n) - n // 2) for name, n in zip(self.pw_dims, self.n_a)
            )
        ]

        # Cartesian components of each G vector, G_c = sum_i b[i, c] * pwka[i]
        self.pwk = [
            sum(self.b[i, c] * self.pwka[i] for i in range(self.n_dims))
            for c in range(self.n_dims)
        ]

        axes = list(range(-self.n_dims, 0))
        self.fV = xr.apply_ufunc(
            lambda arr: fftshift(fftn(arr, axes=axes, norm="forward"), axes=axes),
            self.potential.V,
            input_core_dims=[self.spatial_dims],
            output_core_dims=[self.pw_dims],
        )
        # assign_coords returns a new object, so the result has to be kept: without it fV carries no
        # labels at all and the .loc lookup in set_cutoff silently degrades to positional indexing
        self.fV = self.fV.assign_coords(
            {name: arr for name, arr in zip(self.pw_dims, self.pwka)}
        )

    def set_cutoff(self, E_lim: float):
        """select the wavevectors under an energy cut off and construct an index lookup table to find the terms V_{G-G'}

        Args:
            E_lim (_type_): The energy cut-off, only wavevectors G with alpha*|G|² < E_lim will not be masked.
            It is stored on the solver, so calling this again re-truncates the basis and keeps the repr honest.
        """

        self.E_lim = E_lim
        G2 = sum(pwk**2 for pwk in self.pwk)
        mask = xr.where(self.alpha * G2 < E_lim, 1, 0).transpose(*self.pw_dims)

        # One index array per axis, giving the position of each retained G in the transform
        ndx = np.where(mask)
        index = xr.DataArray(
            list(ndx),
            coords={"ij": np.arange(self.n_dims), "g": np.arange(len(ndx[0]))},
        )
        # Indexers taken with drop=True: keeping the scalar 'ij' would attach it to the coords built
        # below, and kindex could then no longer be indexed along 'ij' at all
        picks = tuple(index.isel(ij=i, drop=True) for i in range(self.n_dims))
        self.kindex = index.assign_coords(
            {  # The position in k-space (coords and cartesian) of each vector G
                name: pwk.transpose(*self.pw_dims)[picks].drop_vars(
                    self.pw_dims, errors="ignore"
                )
                for name, pwk in zip(self.pwk_names, self.pwk)
            }
        )

        self.nGs = len(ndx[0])

        center = xr.DataArray(
            [n // 2 for n in self.n_a], coords={"ij": np.arange(self.n_dims)}
        )
        sizes = xr.DataArray(self.n_a, coords={"ij": np.arange(self.n_dims)})

        # V_{G-G'} of a potential sampled on a discrete grid is periodic in the transform index, so the
        # difference is wrapped back into the array. Without this, a cutoff large enough to make the
        # G index span exceed the grid simply runs off the end of fV.
        raw = index - index.rename({"g": "gp"}) + center
        self.connect = raw % sizes

        # The potential part of the matrix M_{GG'}. 'connect' holds positions in the transform, not
        # the centered pwka labels, so this is deliberately isel and not loc.
        self.matV = self.fV.isel(
            {name: self.connect.sel(ij=i) for i, name in enumerate(self.pw_dims)}
        )

        self.check_aliasing(raw, sizes)

    def check_aliasing(self, raw: xr.DataArray, sizes: xr.DataArray, tol: float = 1e-6):
        """Warn if G-G' had to be wrapped for a potential that its own grid does not resolve.

        Wrapping stands on the assumption that the potential is band-limited, so that the components
        beyond the sampled range, which the wrap replaces, are zero. That holds for a potential built
        from a handful of plane waves, as plane-wave expansion assumes, and there the wrap changes the
        spectrum by nothing at all. It stops holding once the transform has not decayed by the edge of
        the grid, and then the folded-in terms are genuinely spurious. Note that the wrap merely has to
        occur for that to bite, so the test is on the potential's sampling, not on the wrap itself.

        Args:
            raw (xr.DataArray): The unwrapped G-G' positions, before the modulo.
            sizes (xr.DataArray): The size of the transform along each axis.
            tol (float, optional): How large the transform may still be at the edge of the grid,
            relative to its largest component, before it is reported. Defaults to 1e-6.
        """
        if not bool(((raw < 0) | (raw >= sizes)).any()):
            return

        largest = float(abs(self.fV).max())
        # The outermost frequency shell: if the potential is resolved, the transform has died out here
        edge = max(
            float(abs(self.fV.isel({name: [0, n - 1]})).max())
            for name, n in zip(self.pw_dims, self.n_a)
        )
        if largest > 0 and edge > tol * largest:
            warnings.warn(
                f"The cut-off E_lim = {self.E_lim:.3f} makes G-G' reach past the potential's "
                f"{'x'.join(str(n) for n in self.n_a)} grid, and that grid does not resolve the "
                f"potential: its transform is still {edge / largest:.1e} of its peak at the edge. "
                "The terms folded back in are therefore aliased rather than negligible. Raise the "
                "potential's resolution, or lower E_lim.",
                stacklevel=3,
            )

    def diagonalize(
        self, mat: np.ndarray, n_eigva: int, v0: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract the lowest n_eigva eigenpairs of a central-equation matrix.

        The matrix is dense and, for the basis sizes plane wave expansion is usually run at, small.
        ARPACK is built for large sparse operators and spends well over a hundred matrix-vector
        products per solve here, so a direct dense factorization asking only for the bottom of the
        spectrum wins by a wide margin. It stops winning once the basis grows past 'dense_limit',
        where the cubic cost of the factorization overtakes the iteration count.

        Args:
            mat (np.ndarray): The central equation matrix, assumed hermitian.
            n_eigva (int): The number of eigenpairs to extract.
            v0 (np.ndarray, optional): Starting vector, only used by the sparse path. Defaults to None.

        Returns:
            tuple[np.ndarray, np.ndarray]: The eigenvalues and the eigenvectors, as columns.

        Raises:
            ValueError: If more eigenvalues are asked for than the basis can provide.
        """
        if n_eigva > self.nGs:
            raise ValueError(
                f"Asked for {n_eigva} eigenvalues but the plane wave basis only holds {self.nGs} "
                "vectors. Raise E_lim, or ask for fewer bands."
            )

        if self.nGs <= self.dense_limit:
            return eigh(mat, subset_by_index=[0, n_eigva - 1])
        return eigsh(mat, k=n_eigva, v0=v0, which="SA")

    def compute_kinetic(self, k: list[float]) -> np.ndarray:
        """Compute the total kinetic operator.

        The operator is diagonal in the plane wave basis, so only its diagonal is returned: building
        the full (nGs, nGs) matrix just to add it to another one costs two square allocations per
        parameter point, which is no longer negligible now that the diagonalization itself is fast.

        Args:
            k (list[float]): The k vector components, one per axis. Default to zero along every axis.

        Returns:
            np.ndarray: The (nGs,) diagonal of the kinetic operator.
        """

        return (
            sum(
                (self.kindex.coords[name] + k[c]) ** 2
                for c, name in enumerate(self.pwk_names)
            )
            * self.alpha
        ).values

    def set_reciprocal_space(self, k: list[float | xr.DataArray]):
        """Add the reciprocal space to the list of coordinates.

        Args:
            k (list[float | xr.DataArray]): The k vector components, one per axis. Each component can be
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

        k_names = [f"k{c}" for c in self.coord_names]
        k_arrays = []
        for name, ki in zip(k_names, k):
            if isinstance(ki, (float, int)):
                k_arrays += [
                    xr.DataArray(
                        np.array([ki]), coords={name: np.array([ki])}, dims=name
                    )
                ]
            else:
                k_arrays += [xr.DataArray(ki, coords={name: ki}, dims=name)]

        self.set_reciprocal_space(k_arrays)

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

    def initialize_eigve(self, n_eigva: int) -> xr.DataArray:
        """Initialize the array containing the eigenvectors, they are represented in the plane wave basis.

        Args:
            n_eigva (int): The number of eigenvalues to compute
        Returns:
            xr.DataArray: An empty DataArray with the proper shape.
        """
        eigve_coords = [coord[1] for coord in self.allcoords.values()]

        eigve_coords += [self.kindex.coords["g"]]

        eigve_coords += [
            xr.DataArray(np.arange(n_eigva), coords={"band": np.arange(n_eigva)})
        ]

        eigve = empty_from_coords(eigve_coords, complex, "eigve_pw")

        # Carry the cartesian components of every G along, so compute_u can rebuild the profile later
        return eigve.assign_coords(
            {name: self.kindex.coords[name] for name in self.pwk_names}
        )

    def compute_mat(
        self,
        potential_sel: dict,
        reciprocal_sel: dict,
    ) -> np.ndarray:
        """Construct the central equation matrix for the given parameter selection

        Args:
            potential_sel (dict): The parameters selection for the potential.
            reciprocal_sel (dict): The position is k-space.

        Returns:
            np.ndarray
        """

        # Selecting only one value for each potential dimensions, the selection will be empty if there is no potential dimensions

        # The potential is a diagonal matrix, which we stored as a data array.
        potential_matrix = self.matV.sel(potential_sel).data

        # Each k component may be driven by its own subset of the reciprocal dimensions
        k = []
        for ki in self.k:
            if isinstance(ki, xr.DataArray):
                sel = {
                    key: value
                    for key, value in reciprocal_sel.items()
                    if key in ki.dims
                }
                k += [float(ki.sel(sel).data)]
            else:
                k += [ki]
        # Transposed into V_{G'-G}, then the kinetic term added straight onto the diagonal
        mat = potential_matrix.transpose().copy()
        mat[np.diag_indices_from(mat)] += self.compute_kinetic(k)
        return mat

    def solve(
        self,
        n_eigva: int,
        parallel:bool = False,
        n_cores: int = -1,
        verbose:bool = True
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Solve the central equation.

        Args:
            n_eigva (int): The number of eigenvalues to compute.
            parallel (bool, optional): Wheter to parallelise the solver with joblib. Default to False.
            n_cores (int, optional): The number of cores to use for the solver, set to -1 to use all cores available. default to -1.
            verbose (bool, optional): Wheter to inform the user of the solver progress. Default to True.
        Returns:
            tuple[xr.DataArray]: the eigenvalues and the eigenvectors.
        """

        # Create empty DataArrays to store the eigenvalues and vectors
        eigva = self.initialize_eigva(n_eigva)
        eigve = self.initialize_eigve(n_eigva)

        # Create lists of dimensions. The dimensions are separated by type, for efficient Hamiltonian matrix construction.
        potential_dims = [
            dim for dim in self.allcoords if self.allcoords[dim][0] == "potential"
        ]
        reciprocal_dims = [
            dim for dim in self.allcoords if self.allcoords[dim][0] == "reciprocal"
        ]

        # Intializing indexes to loop over.
        potential_index, reciprocal_index = (
            [()],
            [(0, 0)],
        )

        # Stacking the dimensions of each type and replacing the associated index. If there is no dimensions of a given type,
        # the corresponding index is kept as the default defined above
        if len(potential_dims) > 0:
            eigva = eigva.stack(potdims=potential_dims)
            potential_index = eigva.potdims.to_index()
            eigve = eigve.stack(potdims=potential_dims)
        if len(reciprocal_dims) > 0:
            eigva = eigva.stack(recdims=reciprocal_dims)
            reciprocal_index = eigva.recdims.to_index()
            eigve = eigve.stack(recdims=reciprocal_dims)

        # The total number of matrix to diagonalize, used for the progress bar
        n_tot = len(eigva.sel(band=0).data.reshape(-1))

        # Initializing the progress bar
        sels = []
        pot_sels = []
        rec_sels = []
        # Looping over everything
        for pots in potential_index:
            for recs in reciprocal_index:
                # Selecting only one value for each potential dimensions, the selection will be empty if there is no potential dimensions
                potential_sel = (
                    dict(zip(potential_index.names, pots))
                    if hasattr(potential_index, "names")
                    else {}
                )
                reciprocal_sel = (
                    dict(zip(reciprocal_index.names, recs))
                    if hasattr(reciprocal_index, "names")
                    else {}
                )

                pot_sels += [potential_sel]
                rec_sels += [reciprocal_sel]
                
                sels += [
                    {
                        **potential_sel,
                        **reciprocal_sel,
                    }
                ]

        # A first solve, whose leading vector seeds the sparse path's iterations
        e, X = self.diagonalize(self.compute_mat(potential_sel, reciprocal_sel), n_eigva)

        def x(p_sel, r_sel):
            mat = self.compute_mat(p_sel, r_sel)
            e, eigv = self.diagonalize(mat, n_eigva, v0=X[:, 0])
            eigv *= np.exp(-1j*np.angle(eigv[self.nGs//2]))
            return e, eigv

        args = list(zip(pot_sels, rec_sels))
        # The basis size used to be announced in a banner above the bar; it rides in the label
        # instead, where it stays attached to the bar it describes.
        desc = f"Diagonalizing ({self.nGs} plane waves)"

        if parallel:
            results = parallel_map(
                x,
                args,
                n_jobs=min(n_cores, n_tot),
                desc=desc,
                unit="matrix",
                verbose=verbose,
            )
        else:
            results = [
                x(p_sel, r_sel)
                for p_sel, r_sel in bar(
                    args, desc=desc, unit="matrix", verbose=verbose
                )
            ]

        for i in range(n_tot):
            eigvals, eigvecs = results[i][0], results[i][1]

            idx = eigvals.argsort()
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]

            eigva.loc[sels[i]] = eigvals
            eigve.loc[sels[i]] = eigvecs

        if len(potential_dims) > 0:
            eigva = eigva.unstack(dim="potdims")
            eigve = eigve.unstack(dim="potdims")
        if len(reciprocal_dims) > 0:
            eigva = eigva.unstack(dim="recdims")
            eigve = eigve.unstack(dim="recdims")

        return eigva.squeeze(), eigve.squeeze()


    def normalize(self, eigve: xr.DataArray, norm:float = 1)-> xr.DataArray:
        """Normalize the eigenvector array to a specified value in real-space units.

        Args:
            eigve (xr.DataArray): The eigenvector array
            norm (float, optional): The norm of the array. Defaults to 1.

        Returns:
            xr.DataArray
        """
        dims = list(self.spatial_dims)
        if "field" in eigve.dims:
            dims += ["field"]
        normed = eigve / (abs(eigve)**2).sum(dims)**0.5
        return normed * (norm / self.potential.get_dS())**0.5

    def grid_origin(self) -> list[float]:
        """Return the cartesian position of the potential's first grid point, one component per axis.

        The plane wave coefficients are the discrete transform of the potential sampled from that
        corner, so they describe the mode as a function of r - r0 rather than of r itself.

        Returns:
            list[float]: The components of r0.
        """
        first = {d: 0 for d in self.spatial_dims}
        return [float(coord.isel(first)) for coord in self.potential.coords]

    def compute_u_fft(self, eigve: xr.DataArray) -> xr.DataArray:
        """Rebuild the mode profiles on the potential's own grid with a single inverse transform.

        The G vectors sit on a regular grid and b_i.a_j = 2.pi.delta_ij, so G.(r - r0) collapses to
        2.pi.sum_i n_i m_i / N_i for grid index m, and the sum over G is then exactly what fftn
        computes. That turns an O(n_G . n_r) sum into one O(n_r log n_r) transform. Only valid on the
        potential's own grid, since it is the regularity of that grid the transform relies on.

        Args:
            eigve (xr.DataArray): The eigenvector in the plane wave basis.

        Returns:
            xr.DataArray: The mode profiles, unnormalized.
        """
        other_dims = [d for d in eigve.dims if d != "g"]
        source = eigve.transpose("g", *other_dims)

        packed = np.zeros(
            (*self.n_a, *[source.sizes[d] for d in other_dims]), dtype=complex
        )
        # kindex holds positions in the fftshifted transform; ifftshift puts them back in fft order
        packed[
            tuple(self.kindex.isel(ij=i).values for i in range(self.n_dims))
        ] = source.values

        axes = tuple(range(self.n_dims))
        u = fftn(ifftshift(packed, axes=axes), axes=axes)

        return xr.DataArray(
            u,
            dims=[*self.spatial_dims, *other_dims],
            coords={
                # The cartesian coords come along too, the plotting functions read them
                **{
                    name: self.potential.V.coords[name]
                    for name in [*self.spatial_dims, *self.coord_names]
                },
                **{d: source.coords[d] for d in other_dims if d in source.coords},
            },
        )

    def compute_u(
        self,
        eigve: xr.DataArray,
        coords: list[xr.DataArray] | None = None,
        vectorized:bool = False,
        verbose: bool = True,
    ) -> xr.DataArray:
        """Compute the spatial shape of the eigenvectors from their plane-wave expression

        Args:
            eigve (xr.DataArray): The eigenvector in the plane wave basis
            coords (list[xr.DataArray], optional): The cartesian grids over which to sample the eigenvector,
            one per axis. If None, the grids of the potential object are used, and the sum over reciprocal
            vectors is then evaluated as a single inverse FFT instead of term by term. Defaults to None.
            vectorized (bool, optional): Wheter to sum over reciprocal vectors all at once or sequencially. The fully vectorized sum can be slow if the
            resulting matrix is too large. Only consulted when 'coords' is given, since the FFT route
            replaces both. Defaults to False.
            verbose (bool, optional): Whether to plot a progress bar over the reciprocal vectors.
            Only relevant for the sequential sum. Defaults to True.

        Returns:
            xr.DataArray
        """
        if coords is None:
            return self.normalize(self.compute_u_fft(eigve))

        # Every axis is measured from the same corner the transform was taken from. Offsetting only
        # some of them would translate the mode along the others and misplace it in the cell.
        terms = list(zip(self.pwk_names, coords, self.grid_origin()))

        def phase(sel: dict | None = None) -> xr.DataArray:
            """G.(r - r0), for every G at once, or for the single one picked by sel. Kept lazy so the
            sequential branch never has to build the full (g, space) array it exists to avoid."""
            return sum(
                (eigve.coords[name] if sel is None else eigve.coords[name][sel])
                * (coord - origin)
                for name, coord, origin in terms
            )

        # The sign follows from the transpose in compute_mat, which builds the matrix from V_{G'-G};
        # it is the convention that reproduces the FDSolver eigenvectors.
        if vectorized:
            u = (eigve * np.exp(-1j * phase())).sum('g')
        else:
            u = 0
            for ig in bar(
                range(eigve.sizes['g']),
                desc="Summing bands",
                unit="band",
                verbose=verbose,
            ):
                u += eigve[{'g':ig}] * np.exp(-1j * phase({'g':ig}))

        return self.normalize(u)
    
    


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from bloch_schrodinger.plotting import plot_eigenvector

    
    lam = 2
    kl = 2*np.pi / lam

    alpha = 0
    epsilon = 1
    eta = 1

    theta = np.pi/2 + np.pi/21
    # theta = np.pi/2

    a1 = np.array([1, 0])*2**0.5/2 * lam
    a2 = np.array([0, 1])*2**0.5/2 * lam

    b1 = a2 * np.pi * 2 / lam**2
    b2 = a1 * np.pi * 2 / lam**2

    klim = kl
    na1 = 128
    na2 = 128

    V0 = 20
    
    checker = Potential(
        unitvecs = [a1, a2],
        resolution = (na1, na2),
        v0 = 0,
    )

    xmy = (checker.x - checker.y)/2**0.5 - 1/4 * lam
    xpy = (checker.x + checker.y)/2**0.5

    checker.set(
        -V0/4 * abs(
            eta * ( np.exp(1j * kl * (xmy)) + epsilon * np.exp(-1j * kl * (xmy))) +
            np.exp(1j * theta) * (np.exp(1j * kl * xpy) + epsilon * np.exp(-1j * kl * xpy))
        )**2
    )
    
    
    pw = PWSolver(checker, 1/2, 500)
    
    eigva, eigve = pw.solve(2)

    u = pw.compute_u(eigve)
    
    plot_eigenvector(
        [[abs(u)**2]], [[checker]], [['amplitude']]
    )
    plt.show()
