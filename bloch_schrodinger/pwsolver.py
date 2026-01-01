import numpy as np
import xarray as xr
from typing import Union
from bloch_schrodinger.potential import Potential
from tqdm import tqdm, trange
from scipy.sparse.linalg import eigsh
import warnings
from joblib import Parallel, delayed
from numpy.linalg import inv
from scipy.fft import fftn, fftshift


def real(arr: xr.DataArray) -> xr.DataArray:
    return xr.apply_ufunc(np.real, arr)


def imag(arr: xr.DataArray) -> xr.DataArray:
    return xr.apply_ufunc(np.imag, arr)


def check_name(name: str):
    """Check wether the name is a valid one. and raises an error if not.

    Args:
        name (str): The name to check

    Raises:
        ValueError: If the name is forbidden
    """

    forbidden_names = [
        "field",
        "band",
        "a1",
        "a2",
        "x",
        "y",
        "dx",
        "dy",
        "dx2",
        "dy2",
        "lap",
    ]
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
        E_lim: float = None
    ):
        """Instantiate the solver.

        Args:
            potential (Potential): The potential field.
            alpha (float): The kinetic energy coefficient hbar²/2m.
            Elim (float): The energy cut-off.
        """

        self.potential = potential
        self.potential.V = self.potential.V.transpose(..., "a1", "a2")
        self.alpha = alpha

        # storing all parameter coordinates from potentials and alphas. The final solver will run on all these dimensions.
        self.allcoords = {}
        coords_pot = {
            dim: ["potential", potential.V.coords[dim]]
            for dim in potential.V.dims
            if dim not in ["a1", "a2"]
        }
        self.allcoords.update(coords_pot)

        self.kx = 0  # The solver assumes that kx = 0 if not specified otherwise
        self.ky = 0  # The solver assumes that ky = 0 if not specified otherwise

        self.a1_coord = potential.V.coords["a1"]
        self.a2_coord = potential.V.coords["a2"]
        self.a1 = potential.a1  # The first lattice vector
        self.a2 = potential.a2  # The second lattice vector
        self.e1 = self.a1 / (self.a1 @ self.a1) ** 0.5  # normalized lattice vector
        self.e2 = self.a2 / (self.a2 @ self.a2) ** 0.5  # normalized lattice vector
        self.na1 = potential.V.sizes["a1"]  # discretization along a1
        self.na2 = potential.V.sizes["a2"]  # discretization along a2
        self.np = self.na1 * self.na2  # Number of mesh sampling points

        # length steps along a1 and a2
        self.da1 = (
            float(abs(potential.V.a1[1] - potential.V.a1[0]))
            * (self.a1 @ self.a1) ** 0.5
        )  # smallest increment of length along a1
        self.da2 = (
            float(abs(potential.V.a2[1] - potential.V.a2[0]))
            * (self.a2 @ self.a2) ** 0.5
        )  # smallest increment of length along a2

        self.compute_b()
        self.compute_fV()
        
        self.E_lim = E_lim if E_lim is not None else (potential.V.max()-potential.V.min()).item()*2
        self.set_cutoff(self.E_lim)
        if E_lim is None:
            print(f"Energy cut-off not specified and automatically set to {self.E_lim:.3f}, expanding on {self.nGs} vectors")
        

    def __repr__(self) -> str:
        return f"Plane wave Solver object \n E_lim = {self.E_lim:.3f} \n Expanded on {self.nGs} plane waves"

    def compute_b(self):
        """Compute the reciprocal vectors of the unit cell"""
        M = np.array([self.a1, self.a2])
        self.b1, self.b2 = inv(M)[:, 0] * 2 * np.pi, inv(M)[:, 1] * 2 * np.pi

    def compute_fV(self):
        """Compute the fourier transform of the potential matrix"""

        ka1 = np.arange(self.na1) - (self.na1) // 2
        ka2 = np.arange(self.na2) - (self.na2) // 2

        self.pwka1 = xr.DataArray(ka1, coords={"pwka1": ka1})
        self.pwka2 = xr.DataArray(ka2, coords={"pwka2": ka2})

        self.pwkx = self.b1[0] * self.pwka1 + self.b2[0] * self.pwka2
        self.pwky = self.b1[1] * self.pwka1 + self.b2[1] * self.pwka2

        self.fV = xr.apply_ufunc(
            lambda arr: fftshift(
                fftn(arr, axes=[-2, -1], norm="forward"), axes=[-2, -1]
            ),
            self.potential.V,
            input_core_dims=[["a1", "a2"]],
            output_core_dims=[["pwka1", "pwka2"]],
        )
        self.fV.assign_coords({"pwka1": self.pwka1, "pwka2": self.pwka2})

    def set_cutoff(self, E_lim: float):
        """select the wavevectors under an energy cut off and construct an index lookup table to find the terms V_{G-G'}

        Args:
            E_lim (_type_): The energy cut-off, only wavevectors G with alpha*|G|² < E_lim will not be masked.
        """

        mask = xr.where(self.alpha * (self.pwkx**2 + self.pwky**2) < E_lim, 1, 0)

        indx, jndx = np.where(mask)
        index = xr.DataArray(
            [indx, jndx], coords={"ij": np.arange(2), "g": np.arange(len(indx))}
        )

        self.kindex = index.assign_coords(
            {  # The position in k-space (coords and cartesian) of each vector G
                "pwkx": self.pwkx[index.sel(ij=0), index.sel(ij=1)],
                "pwky": self.pwky[index.sel(ij=0), index.sel(ij=1)],
            }
        )

        self.nGs = len(indx)

        center = xr.DataArray(
            [self.na1 // 2, self.na2 // 2], coords={"ij": np.arange(2)}
        )

        connect = index - index.rename({"g": "gp"}) + center

        self.matV = self.fV.loc[  # The potential part of the matrix M_{GG'}
            {"pwka1": connect.sel(ij=0), "pwka2": connect.sel(ij=1)}
        ]

    def compute_kinetic(self, k: tuple[float, float]) -> np.ndarray:
        """Compute the total kinetic operator.

        Args:
            k (tuple[float]): The k vector. default to (0,0)
        """

        diag = (
            (self.kindex.pwkx + k[0]) ** 2 + (self.kindex.pwky + k[1]) ** 2
        ) * self.alpha
        return np.diag(diag)

    def set_reciprocal_space(
        self, kx: Union[float, xr.DataArray], ky: Union[float, xr.DataArray]
    ):
        """Add the reciprocal space to the list of coordinates.

        Args:
            kx (Union[float,xr.DataArray]): kx coordinate, can be a single float, a 1D xarray coordinate or even a multidimensional coordinate.
            ky (Union[float,xr.DataArray]): ky coordinate, can be a single float, a 1D xarray coordinate or even a multidimensional coordinate.
        """
        if isinstance(kx, xr.DataArray):
            self.allcoords.update(
                {dim: ["reciprocal", kx.coords[dim]] for dim in kx.dims}
            )
        if isinstance(ky, xr.DataArray):
            self.allcoords.update(
                {dim: ["reciprocal", ky.coords[dim]] for dim in ky.dims}
            )

        self.kx = kx
        self.ky = ky

    def create_reciprocal_grid(
        self, kx: Union[float, np.ndarray] = 0, ky: Union[float, np.ndarray] = 0
    ):
        """Create the k-space grid on which the eigenvalues and vectors will be computed

        Args:
            kx (float or np.ndarray): The values of kx for the grid points. default to 0.
            ky (float or np.ndarray): The values of ky for the grid points. default to 0.
        """
        if isinstance(kx, float) or isinstance(kx, int):
            kx = xr.DataArray(np.array([kx]), coords={"kx": np.array([kx])}, dims="kx")
        else:
            kx = xr.DataArray(kx, coords={"kx": kx}, dims="kx")

        if isinstance(ky, float) or isinstance(ky, int):
            ky = xr.DataArray(np.array([ky]), coords={"ky": np.array([ky])}, dims="ky")
        else:
            ky = xr.DataArray(ky, coords={"ky": ky}, dims="ky")

        self.set_reciprocal_space(kx, ky)

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

        all_coords = {}
        for arr in eigva_coords:
            for d in arr.dims:
                # prefer coords over raw range
                if d in arr.coords:
                    coord = arr.coords[d]
                else:
                    coord = np.arange(arr.sizes[d])
                # if already seen, ensure consistent
                if d not in all_coords:
                    all_coords[d] = coord

        # 2. Compute the shape
        shape = [
            all_coords[d].sizes[d]
            if isinstance(all_coords[d], xr.DataArray)
            else len(all_coords[d])
            for d in all_coords
        ]

        # 3. Create the zero-filled DataArray
        eigva = np.zeros(shape)
        eigva = xr.DataArray(
            eigva, dims=list(all_coords.keys()), coords=all_coords, name="eigva"
        )

        return eigva

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

        all_coords = {}
        for arr in eigve_coords:
            for d in arr.dims:
                # prefer coords over raw range
                if d in arr.coords:
                    coord = arr.coords[d]
                else:
                    coord = np.arange(arr.sizes[d])
                # if already seen, ensure consistent
                if d not in all_coords:
                    all_coords[d] = coord

        # 2. Compute the shape
        shape = [
            all_coords[d].sizes[d]
            if isinstance(all_coords[d], xr.DataArray)
            else len(all_coords[d])
            for d in all_coords
        ]

        # 3. Create the zero-filled DataArray
        eigve = np.zeros(shape, dtype=complex)
        eigve = xr.DataArray(
            eigve, dims=list(all_coords.keys()), coords=all_coords, name="eigve_pw"
        )

        eigve = eigve.assign_coords(
            {"pwkx": self.kindex.pwkx, "pwky": self.kindex.pwky}
        )

        return eigve

    def solve(self, n_eigva: int) -> tuple[xr.DataArray, xr.DataArray]:
        """Solve the eigenvalue problem at every points of the parameter space, using the scipy.sparse.eigsh function.
        The eigenvalues and vectors are returned as properly shaped DataArrays.

        Args:
            n_eigva (int): The number of eigenvalues to compute.
        Returns:
            tuple[xr.DataArray]: the eigenvalues and the eigenvectors if keep_vecs is True.
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

        # Initializing the vector guess. The solver works better with a good guess for the lowest eigenvector
        X = np.random.rand(self.nGs, n_eigva)

        # Initializing the progress bar
        with tqdm(total=n_tot) as pbar:
            count = 0

            # Looping over first the potential dimensions, then the alpha dimensions, then the reciprocal dimensions and finally the coupling dimensions.
            for pots in potential_index:
                # Selecting only one value for each potential dimensions, the selection will be empty if there is no potential dimensions
                potential_sel = (
                    dict(zip(potential_index.names, pots))
                    if hasattr(potential_index, "names")
                    else {}
                )
                # The potential is a diagonal matrix, which we stored as a data array.
                potential_matrix = self.matV.sel(potential_sel).data

                for recs in reciprocal_index:
                    reciprocal_sel = (
                        dict(zip(reciprocal_index.names, recs))
                        if hasattr(reciprocal_index, "names")
                        else {}
                    )
                    if isinstance(self.kx, xr.DataArray):
                        kxsel = {
                            key: value
                            for key, value in reciprocal_sel.items()
                            if key in self.kx.dims
                        }
                        kx = float(self.kx.sel(kxsel).data)
                    else:
                        kx = self.kx

                    if isinstance(self.ky, xr.DataArray):
                        kysel = {
                            key: value
                            for key, value in reciprocal_sel.items()
                            if key in self.ky.dims
                        }
                        ky = float(self.ky.sel(kysel).data)
                    else:
                        ky = self.ky
                    kinetic_matrix = self.compute_kinetic([kx, ky])

                    total_sel = {
                        **potential_sel,
                        **reciprocal_sel,
                    }  # aggregating all the values of each selections
                    mat = kinetic_matrix + potential_matrix

                    with warnings.catch_warnings(record=True):
                        # "always" makes them get appended to caught so they are not printed
                        warnings.simplefilter("always")
                        eigvals, eigvecs = eigsh(mat, k=n_eigva, v0=X[:, 0], which="SA")
                    idx = eigvals.argsort()
                    eigvals = eigvals[idx]
                    eigvecs = eigvecs[:, idx]

                    X = eigvecs

                    eigva.loc[total_sel] = eigvals
                    eigve.loc[total_sel] = eigvecs
                    pbar.update(1)
                    count += 1

        if len(potential_dims) > 0:
            eigva = eigva.unstack(dim="potdims")
            eigve = eigve.unstack(dim="potdims")
        if len(reciprocal_dims) > 0:
            eigva = eigva.unstack(dim="recdims")
            eigve = eigve.unstack(dim="recdims")

        return eigva.squeeze(), eigve.squeeze()

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

        if isinstance(self.kx, xr.DataArray):
            kxsel = {
                key: value
                for key, value in reciprocal_sel.items()
                if key in self.kx.dims
            }
            kx = float(self.kx.sel(kxsel).data)
        else:
            kx = self.kx

        if isinstance(self.ky, xr.DataArray):
            kysel = {
                key: value
                for key, value in reciprocal_sel.items()
                if key in self.ky.dims
            }
            ky = float(self.ky.sel(kysel).data)
        else:
            ky = self.ky
        kinetic_matrix = self.compute_kinetic([kx, ky])

        mat = potential_matrix + kinetic_matrix
        return mat

    def parallel_solve(
        self,
        n_eigva: int,
        n_cores: int = -1,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Parallel version of the solve function. It generates first every matrix to solve so be mindful of memory constraints.

        Args:
            n_eigva (int): The number of eigenvalues to compute.
            n_cores (int, optional): The number of cores to use for the solver, set to -1 to use all cores available. default to -1.
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
        print("Creating matrices")
        mat_list = []
        sels = []
        with tqdm(total=n_tot) as pbar:
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

                    mat = self.compute_mat(potential_sel, reciprocal_sel)

                    mat_list += [mat]
                    sels += [
                        {
                            **potential_sel,
                            **reciprocal_sel,
                        }
                    ]
                    pbar.update(1)

        e, X = eigsh(mat, k=n_eigva, which="SA")

        def x(y):
            return eigsh(y, k=n_eigva, v0=X[:, 0], which="SA")

        print("Performing the diagonalization...")
        parallel = Parallel(n_jobs=n_cores, return_as="list", verbose=5)
        results = parallel(delayed(x)(y) for y in mat_list)
        # print(results)

        print("storing the results")
        with tqdm(total=n_tot) as pbar:
            for i in range(n_tot):
                eigvals, eigvecs = results[i][0], results[i][1]

                idx = eigvals.argsort()
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]

                eigva.loc[sels[i]] = eigvals
                eigve.loc[sels[i]] = eigvecs
                pbar.update(1)

        if len(potential_dims) > 0:
            eigva = eigva.unstack(dim="potdims")
            eigve = eigve.unstack(dim="potdims")
        if len(reciprocal_dims) > 0:
            eigva = eigva.unstack(dim="recdims")
            eigve = eigve.unstack(dim="recdims")

        return eigva.squeeze(), eigve.squeeze()

    def compute_u(
        self, 
        eigve: xr.DataArray, 
        x: xr.DataArray = None, 
        y: xr.DataArray = None,
        vectorized:bool = False
    ) -> xr.DataArray:
        """Compute the spatial shape of the eigenvectors from their plane-wave expression

        Args:
            eigve (xr.DataArray): The eigenvector in the plane wave basis
            x (xr.DataArray, optional): The x-grid over which to sample the eigenvector, if None, then the grid from the potential is chosen. Defaults to None.
            y (xr.DataArray, optional): The y-grid over which to sample the eigenvector, if None, then the grid from the potential is chosen. Defaults to None.
            vectorized (bool, optional): Wheter to sum over reciprocal vectors all at once or sequencially. The fully vectorized sum can be slow if the 
            resulting matrix is too large. Defaults to False.

        Returns:
            xr.DataArray
        """
        if x is None:
            x = self.potential.x - self.potential.a1[0]
        if y is None:
            y = self.potential.y
        
        if vectorized:
            u = (
                eigve * np.exp(1j * (eigve.pwkx * x + eigve.pwky * y))
            ).sum('g')
            print(end = '\r')
        else:
            print('summing...')
            u = 0
            for ig in trange(eigve.sizes['g']):
                u += eigve[{'g':ig}] * np.exp(1j * (eigve.pwkx[{'g':ig}] * x + eigve.pwky[{'g':ig}] * y))
        
        return u / (abs(u)**2).sum(['a1', 'a2'])**0.5
    
    


if __name__ == "__main__":
    import matplotlib.pyplot as plt  # noqa: E402
    import time as time  # noqa: E402
    from bloch_schrodinger.potential import create_parameter
    from bloch_schrodinger.plotting import plot_cuts, plot_eigenvector
    # First, we define a few constants

    kl = 2
    a = 4 * np.pi / 3**1.5 / kl
    a1 = np.array([3 * a / 2, -(3**0.5) * a / 2])  # 1st lattice vector
    a2 = np.array([3 * a / 2, 3**0.5 * a / 2])  # 2nd lattice vector

    b1 = 3**0.5 * kl * np.array([1, -(3**0.5)]) / 2
    b2 = 3**0.5 * kl * np.array([1, 3**0.5]) / 2

    m = 1
    hbar = 1
    E_r = hbar**2 * kl**2 / 2 / m
    s1 = 20
    s2 = s1 + create_parameter("delta", np.linspace(-0.5, 0.5, 5))

    k1 = kl * np.array([-(3**0.5) / 2, 1 / 2])
    k2 = kl * np.array([3**0.5 / 2, 1 / 2])
    k3 = kl * np.array([0, -1])

    a1s = np.array([-1, 3**0.5]) * 2 * np.pi / 3 / a
    a2s = np.array([1, 3**0.5]) * 2 * np.pi / 3 / a
    K = np.array([0, 4 * np.pi / 3**1.5 / a])

    klim = 1
    na1 = 64
    na2 = 64

    V1 = -s2 * E_r
    V2 = -s1 * E_r

    honeycomb = Potential(
        unitvecs=[a1, a2],
        resolution=(na1, na2),
        v0=50,
    )

    dirs = [
        k1[0] * (honeycomb.x - a1[0]) + k1[1] * honeycomb.y,
        k2[0] * (honeycomb.x - a1[0]) + k2[1] * honeycomb.y,
        k3[0] * (honeycomb.x - a1[0]) + k3[1] * honeycomb.y,
    ]

    for i in range(3):
        honeycomb.add(2 * V1 * np.cos((dirs[i - 1] - dirs[i]) - 2 * np.pi / 3) / 2)
        honeycomb.add(2 * V2 * np.cos((dirs[i - 1] - dirs[i]) + 2 * np.pi / 3) / 2)

    # honeycomb.plot()
    # plt.show()

    alp = 1
    pw = PWSolver(honeycomb, alp)

    pw.create_reciprocal_grid(
        ky=np.linspace(-5, 5, 51),
        kx=0,
    )
    eigva, eigve = pw.parallel_solve(2)

    u = pw.compute_u(eigve, vectorized=False)

    plot_cuts(eigva, "ky")
    plt.show()
    
    plot_eigenvector(
        [[abs(u)**2]], [[honeycomb]], [['amplitude']]
    )
    plt.show()
