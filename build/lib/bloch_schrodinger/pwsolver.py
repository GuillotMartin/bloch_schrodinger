import numpy as np
import xarray as xr
from typing import Union
from bloch_schrodinger.potential import Potential
import scipy.sparse as sps
from tqdm import tqdm
from scipy.sparse.linalg import eigsh
import warnings
from joblib import Parallel, delayed
from numpy.linalg import svd, inv
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
    It only supports scalar equations for now."""

    def __init__(
        self,
        potential: Potential,
        alpha: Union[float, xr.DataArray],
    ):
        """Instantiate the solver.

        Args:
            potential (Potential): The potential field.
            alpha (Union[float, xr.DataArray]): The kinetic energy coefficient hbar²/2m.
        """

        self.potential = potential
        self.potential.V = self.potential.V.transpose(..., 'a1', 'a2')
        self.alpha = alpha
        
        # storing all parameter coordinates from potentials and alphas. The final solver will run on all these dimensions.
        self.allcoords = {}
        coords_pot = {
            dim: ["potential", potential.V.coords[dim]]
            for dim in potential.V.dims
            if dim not in ["a1", "a2"]
        }
        self.allcoords.update(coords_pot)

        if isinstance(alpha, xr.DataArray):
            for dim in alpha.dims:
                check_name(dim)
                coords_alpha = {
                    dim: ["alpha", alpha.coords[dim]] for dim in alpha.dims
                }
                self.allcoords.update(coords_alpha)

        self.kx = 0  # The solver assumes that kx = 0 if not specified otherwise
        self.ky = 0  # The solver assumes that ky = 0 if not specified otherwise

        self.a1_coord = potential.V.coords["a1"]
        self.a2_coord = potential.V.coords["a2"]
        self.a1 = potential.a1  # The first lattice vector
        self.a2 = potential.a2  # The second lattice vector
        self.e1 = self.a1 / (self.a1 @ self.a1) ** 0.5  # normalized lattice vector
        self.e2 = self.a2 / (self.a2 @ self.a2) ** 0.5  # normalized lattice vector
        self.na1 = potential.V.sizes['a1']  # discretization along a1
        self.na2 = potential.V.sizes['a2']  # discretization along a2
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


    def __repr__(self) -> str:
        shape = {dim: len(self.allcoords[dim][1].data) for dim in self.allcoords}
        return f"Plane wave Solver object \n size ({self.np}, {self.np}), with {1} field \n dimensions: {shape}"

    def compute_b(self):
        """Compute the reciprocal vectors of the unit cell
        """
        M = np.array([self.a1, self.a2])
        self.b1, self.b2 = inv(M)[:,0]*2*np.pi, inv(M)[:,1]*2*np.pi
        

    def compute_fV(self):
        """Compute the fourier transform of the 
        """
        
        ka1 = np.arange(self.na1)-(self.na1)//2
        ka2 = np.arange(self.na2)-(self.na2)//2

        self.ka1 = xr.DataArray(
            ka1, 
            coords = {'ka1':ka1}
        )
        self.ka2 = xr.DataArray(
            ka2, 
            coords = {'ka2':ka2}
        )
        
        self.pwkx = self.b1[0] * self.ka1 + self.b2[0] * self.ka2
        self.pwky = self.b1[1] * self.ka1 + self.b2[1] * self.ka2
        
        self.fV = xr.apply_ufunc(
            lambda arr: fftshift(fftn(
                arr, axes = [-2,-1]
            )),
            self.potential.V,
            input_core_dims=[['a1', 'a2']],
            output_core_dims=[['ka1', 'ka2']],
        )
        self.fV.assign_coords({'ka1':self.ka1, 'ka2':self.ka2})
                        
        
    
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

    def initialize_eigve(self, n_eigva: int, stack: bool = True) -> xr.DataArray:
        """Initialize the array containing the flattened eigenvectors

        Args:
            n_eigva (int): The number of eigenvalues to compute
            stack (bool): Wheter to stack the eigenvectors along a single 'component' dimension. Somes children classes don't want that. default to True
        Returns:
            xr.DataArray: An empty DataArray with the proper shape.
        """
        eigve_coords = [coord[1] for coord in self.allcoords.values()]

        eigve_coords += [
            xr.DataArray(np.arange(self.nb), coords={"field": np.arange(self.nb)})
        ]

        eigve_coords += [self.a1_coord, self.a2_coord]

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
            eigve, dims=list(all_coords.keys()), coords=all_coords, name="eigve_flat"
        )

        x = self.a1[0] * eigve.a1 + self.a2[0] * eigve.a2
        y = self.a1[1] * eigve.a1 + self.a2[1] * eigve.a2
        eigve = eigve.assign_coords(
            {
                "x": x,
                "y": y,
            }
        )
        if stack:
            eigve = eigve.stack(component=("field", "a1", "a2")).transpose(
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

    def solve(
        self, n_eigva: int, keep_vecs: bool = "True", **kwargs
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Solve the eigenvalue problem at every points of the parameter space, using the scipy.sparse.eigsh function. This function is painfully parallelizable,
        you are welcome to create your own based on your architecture and needs. The eigenvalues and vectors are returned as properly shaped DataArrays.

        Args:
            n_eigva (int): The number of eigenvalues to compute.
            keep_vecs (bool, optional): Wether to keep the eigenvectors. Defaults to 'True'.
        Returns:
            tuple[xr.DataArray]: the eigenvalues and the eigenvectors if keep_vecs is True.

        Additional kwargs:
            phase0 (tuple[float,float,int]): The position at which to fix the phase of each eigenvector, in a1, a2, field basis. Default to (0.01, 0.01, 0).
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

        # Initializing the vector guess. The solver works better with a good guess for the lowest eigenvector
        X = np.random.rand(self.n, n_eigva)

        # Initializing the progress bar
        pbar = tqdm(total=n_tot)
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
            potdiag = self.potential_data.sel(potential_sel).data
            potential_matrix = sps.diags(potdiag, offsets=0)

            for alphs in alpha_index:
                # Same for the alphas, using the 'make_alpha_list' class method
                alpha_sel = (
                    dict(zip(alpha_index.names, alphs))
                    if hasattr(alpha_index, "names")
                    else {}
                )
                alphas = self.make_alpha_list(alpha_sel)
                self.compute_full_operators(alphas)

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

                    for coups in coupling_index:
                        coupling_sel = (
                            dict(zip(coupling_index.names, coups))
                            if hasattr(coupling_index, "names")
                            else {}
                        )
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
                        self.coupling_context.update(total_sel)
                        for coupling in self.couplings:
                            ham += eval(
                                coupling, {"__builtins__": {}}, self.coupling_context
                            )

                        with warnings.catch_warnings(record=True) as caught:
                            # "always" makes them get appended to caught so they are not printed
                            warnings.simplefilter("always")
                            eigvals, eigvecs = eigsh(
                                ham, k=n_eigva, v0=X[:, 0], which="SM"
                            )
                        idx = eigvals.argsort()
                        eigvals = eigvals[idx]
                        eigvecs = eigvecs[:, idx]

                        X = eigvecs

                        eigva.loc[total_sel] = eigvals
                        if keep_vecs:
                            eigve.loc[total_sel] = eigvecs
                        pbar.update(1)
                        count += 1
        pbar.close()

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

            pos0 = kwargs.get("phase0", (0.01, 0.01, 0))
            sel0 = dict(a1=pos0[0], a2=pos0[1], field=pos0[2])

            eigve = eigve * xr.ufuncs.exp(
                1j * xr.ufuncs.angle(eigve.sel(sel0, method="nearest"))
            )
            x = self.a1[0] * eigve.a1 + self.a2[0] * eigve.a2
            y = self.a1[1] * eigve.a1 + self.a2[1] * eigve.a2
            eigve = eigve.assign_coords(
                {
                    "x": x,
                    "y": y,
                }
            )

        if keep_vecs:
            return eigva.squeeze(), eigve.squeeze()
        else:
            return eigva.squeeze()

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
            reciprocal_sel (dict): The position is k-space.
            coupling_sel (dict): The parameters selection for the additional coupling terms.

        Returns:
            sps.spmatrix: The Hamiltonian.
        """

        # Selecting only one value for each potential dimensions, the selection will be empty if there is no potential dimensions

        # The potential is a diagonal matrix, which we stored as a data array.
        potdiag = self.potential_data.sel(potential_sel).data
        potential_matrix = sps.diags(potdiag, offsets=0)

        # Same for the alphas, using the 'make_alpha_list' class method
        alphas = self.make_alpha_list(alpha_sel)
        self.compute_full_operators(alphas)

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
            **alpha_sel,
            **reciprocal_sel,
            **coupling_sel,
        }  # aggregating all the values of each selections
        ham = (
            kinetic_matrix + potential_matrix
        )  # The initial Hamiltonian contains only the kinetic operator and the potential operator

        # --- Add the coupling terms to the Hamiltonian ---
        self.coupling_context.update(total_sel)
        for coupling in self.couplings:
            ham += eval(coupling, {"__builtins__": {}}, self.coupling_context)

        return ham

    def parallel_solve(
        self, n_eigva: int, keep_vecs: bool = "True", n_cores: int = 8, **kwargs
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Parallel solve the eigenvalue problem at every points of the parameter space, using the scipy.sparse.eigsh function.
        Because this function generates all the hamiltonian first, it uses a lot of memory. Use it for large batches of small problems.

        Args:
            n_eigva (int): The number of eigenvalues to compute.
            n_cores (int, optional): The number of cores to use for the solver, set to -1 to use all cores available. default to 8.
        Returns:
            tuple[xr.DataArray]: the eigenvalues and the eigenvectors if keep_vecs is True.

        Additional kwargs:
            phase0 (tuple[float,float,int]): The position at which to fix the phase of each eigenvector, in a1, a2, field basis. Default to (0, 0, 0).
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
        print("Creating Hamiltonians")
        Ham_list = []
        sels = []
        with tqdm(total=n_tot) as pbar:
            # Looping over everything
            for pots in potential_index:
                for alphs in alpha_index:
                    for recs in reciprocal_index:
                        for coups in coupling_index:
                            # Selecting only one value for each potential dimensions, the selection will be empty if there is no potential dimensions
                            potential_sel = (
                                dict(zip(potential_index.names, pots))
                                if hasattr(potential_index, "names")
                                else {}
                            )
                            alpha_sel = (
                                dict(zip(alpha_index.names, alphs))
                                if hasattr(alpha_index, "names")
                                else {}
                            )
                            reciprocal_sel = (
                                dict(zip(reciprocal_index.names, recs))
                                if hasattr(reciprocal_index, "names")
                                else {}
                            )
                            coupling_sel = (
                                dict(zip(coupling_index.names, coups))
                                if hasattr(coupling_index, "names")
                                else {}
                            )

                            ham = self.create_hamiltonian(
                                potential_sel, alpha_sel, reciprocal_sel, coupling_sel
                            )

                            Ham_list += [ham]
                            sels += [
                                {
                                    **potential_sel,
                                    **alpha_sel,
                                    **reciprocal_sel,
                                    **coupling_sel,
                                }
                            ]
                            pbar.update(1)

        e, X = eigsh(ham, k=n_eigva, which="SM")

        def x(y):
            return eigsh(y, k=n_eigva, v0=X[:, 0], which="SM")

        print("Performing the diagonalization...")
        parallel = Parallel(n_jobs=n_cores, return_as="list", verbose=5)
        results = parallel(delayed(x)(y) for y in Ham_list)
        # print(results)

        print("storing the results")
        with tqdm(total=n_tot) as pbar:
            for i in range(n_tot):
                eigvals, eigvecs = results[i][0], results[i][1]

                idx = eigvals.argsort()
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]

                eigva.loc[sels[i]] = eigvals
                if keep_vecs:
                    eigve.loc[sels[i]] = eigvecs
                pbar.update(1)

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

            pos0 = kwargs.get("phase0", (0, 0, 0))
            sel0 = dict(a1=pos0[0], a2=pos0[1], field=pos0[2])

            eigve = eigve * xr.ufuncs.exp(
                -1j * xr.ufuncs.angle(eigve.sel(sel0, method="nearest"))
            )
            xc = self.a1[0] * eigve.a1 + self.a2[0] * eigve.a2
            yc = self.a1[1] * eigve.a1 + self.a2[1] * eigve.a2
            eigve = eigve.assign_coords(
                {
                    "x": xc,
                    "y": yc,
                }
            )

        if keep_vecs:
            return eigva.squeeze(), eigve.squeeze()
        else:
            return eigva.squeeze()


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt  # noqa: E402
    import time as time # noqa: E402
    from bloch_schrodinger.potential import create_parameter

    # First, we define a few constants

    kl = 2
    a = 4*np.pi / 3**1.5 / kl
    a1 = np.array([3 * a/2, -(3**0.5) * a/2])  # 1st lattice vector
    a2 = np.array([3 * a/2, 3**0.5 * a/2])  # 2nd lattice vector

    b1 = 3**0.5 * kl * np.array([1, -3**0.5])/2
    b2 = 3**0.5 * kl * np.array([1, 3**0.5])/2

    m = 1
    hbar = 1
    E_r = hbar**2 * kl**2 / 2 / m
    s1 = 10
    s2 = s1 + create_parameter('delta', np.linspace(-0.5, 0.5, 3))

    k1 = kl * np.array([-3**0.5 / 2 , 1/2])
    k2 = kl * np.array([3**0.5 / 2 , 1/2])
    k3 = kl * np.array([0,-1])

    a1s = np.array([-1, 3**0.5])*2*np.pi/3/a 
    a2s = np.array([1, 3**0.5])*2*np.pi/3/a
    K = np.array([0, 4*np.pi/3**1.5/a])


    
    klim = 1
    na1 = 100
    na2 = 100
    print(2*np.pi/(3*a/2)*na1)

    V1 = -s2*E_r
    V2 = -s1*E_r

    honeycomb = Potential(
        unitvecs = [a1, a2],
        resolution = (na1, na2),
        v0 = 50,
    )

    dirs = [
        k1[0] * (honeycomb.x - a1[0]) + k1[1] * honeycomb.y,
        k2[0] * (honeycomb.x - a1[0]) + k2[1] * honeycomb.y,
        k3[0] * (honeycomb.x - a1[0]) + k3[1] * honeycomb.y,
    ]

    for i in range(3):
        honeycomb.add(2*V1*np.cos((dirs[i-1]-dirs[i]) - 2*np.pi/3)/2)
        honeycomb.add(2*V2*np.cos((dirs[i-1]-dirs[i]) + 2*np.pi/3)/2)

    # honeycomb.plot()

    alp = create_parameter('alpha', np.linspace(0.8, 1, 2))
    pw = PWSolver(honeycomb, alp)