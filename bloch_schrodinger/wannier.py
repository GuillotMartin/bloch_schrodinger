import numpy as np
import xarray as xr
from typing import Union
from scipy.linalg import expm, fractional_matrix_power
from numpy.linalg import svd, inv
from numpy.random import uniform
from tqdm import tqdm, trange
from joblib import Parallel, delayed
from xarray_einstats.linalg import matmul

from bloch_schrodinger.fdsolver import FDSolver
from bloch_schrodinger.pwsolver import PWSolver
from bloch_schrodinger.potential import Potential, create_parameter

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
            rec_vecs (list[list[float, float]]): The reciprocal vectors of the unit cell, given as [b1, b2] with bi a size-2 list or array.
            resolution (tuple[int, int]): The resolution for the k-space Monkhorst-Pack grid used. 
            method (str, optional): The Bloch-Schrödinger solver to use. Either 'pw' for plane waves or 'fd' for finite differences. Defaults to 'pw'.
        """
        self.potential = Potential
        self.alpha = alpha
        self.method = method
        
        self.b1 = rec_vecs[0]
        self.b2 = rec_vecs[1]
        self.nb1 = resolution[0]
        self.nb2 = resolution[1]
        
        
        self.kb1 = create_parameter("kb1", np.linspace(-1/2, 1/2, self.nb1, endpoint=False)+1/2/self.nb1)
        self.kb2 = create_parameter("kb2", np.linspace(-1/2, 1/2, self.nb2, endpoint=False)+1/2/self.nb2)
        
        self.kx = self.b1[0] * self.kb1 + self.b2[0] * self.kb2
        self.ky = self.b1[1] * self.kb1 + self.b2[1] * self.kb2
        
        self.maxsearch = min(min(self.nb1, self.nb2)//2-2, 10)
        self.compute_stencil()
        
        
    def compute_stencil(self):
        """Creates the weights for the gradient and laplacian operator definitions, see https://arxiv.org/pdf/0708.0650 sec. 3.2 for more details."""

        i0, j0 = self.nb1 // 2, self.nb2 // 2
        kx0, ky0 = self.kx[i0, j0].item(), self.ky[i0, j0].item()

        lim = self.maxsearch
        # Computing the distances in index and cartesian coordinates for a few shells
        I, J, kxs, kys, Dist = [], [], [], [], []  # noqa: E741
        for i in range(-lim, lim + 1):
            for j in range(-lim, lim + 1):
                if not (i == 0 and j == 0):
                    kx1, ky1 = self.kx[i0 + i, j0 + j].item(), self.ky[i0 + i, j0 + j].item()
                    dist = ((kx1 - kx0) ** 2 + (ky1 - ky0) ** 2) ** 0.5
                    Dist += [dist]
                    I += [i]  # noqa: E741
                    J += [j]
                    kxs += [kx1 - kx0]
                    kys += [ky1 - ky0]
                    
        I = np.array(I)  # noqa: E741
        J = np.array(J)
        Dist = np.array(Dist)
        kxs = np.array(kxs)
        kys = np.array(kys)

        # Sorting all the points by ascending distance to the central one
        sorting = np.argsort(Dist)
        I_sorted = I[sorting]
        J_sorted = J[sorting]
        Dist_sorted = Dist[sorting]
        kxs_sorted = kxs[sorting]
        kys_sorted = kys[sorting]

        # Sorting the points by shell number
        shell_distance, start_shell, shell_index, n_in_shell = np.unique(
            Dist_sorted.round(decimals=8),
            return_index=True,
            return_inverse=True,
            return_counts=True,
        )

        # Solve for the weights, keep adding shells until a
        solved = False
        nshell = 1
        while not solved:
            q = np.array([1, 0, 1]) # solving for symmetric weights
            A_js = np.array(
                [
                    [
                        sum(
                            [
                                kxs_sorted[start_shell[s] + m]
                                * kxs_sorted[start_shell[s] + m]
                                for m in range(n_in_shell[s])
                            ]
                        )
                        for s in range(nshell)
                    ],
                    [
                        sum(
                            [
                                kxs_sorted[start_shell[s] + m]
                                * kys_sorted[start_shell[s] + m]
                                for m in range(n_in_shell[s])
                            ]
                        )
                        for s in range(nshell)
                    ],
                    [
                        sum(
                            [
                                kys_sorted[start_shell[s] + m]
                                * kys_sorted[start_shell[s] + m]
                                for m in range(n_in_shell[s])
                            ]
                        )
                        for s in range(nshell)
                    ],
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

        weights = []
        neighbors = []
        neighbors_indexes = []
        for s in range(nshell):
            for m in range(n_in_shell[s]):
                indx = start_shell[s] + m
                neighbors_indexes += [(I_sorted[indx], J_sorted[indx])]
                neighbors += [[kxs_sorted[indx], kys_sorted[indx]]]
                weights += [w[s]]

        self.stencil_size = len(weights)
        self.weights = xr.DataArray(weights, coords={"b": np.arange(self.stencil_size)})
        self.neighbors = xr.DataArray(
            neighbors, coords={"b": np.arange(self.stencil_size), "kxy": [0, 1]}
        )
        self.neighbors_indexes = xr.DataArray(
            neighbors_indexes, coords={"b": np.arange(self.stencil_size), "ij": [0, 1]}
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
        nbands = self.n_wannier # shorter names
        nb = self.stencil_size # shorter names

        M_mnkb = xr.DataArray(
            np.zeros((nbands, nbands, self.nb1, self.nb1, nb), dtype=np.complex128),
            dims=['m','n','kb1','kb2','b']
        )

        M_mnkb.coords["kb1"] = self.kb1
        M_mnkb.coords["kb2"] = self.kb2


        for b_idx in range(nb):
            delta_i = self.neighbors_indexes.sel(b=b_idx, ij=0).item()
            delta_j = self.neighbors_indexes.sel(b=b_idx, ij=1).item()
            
            u_nkpb = u_mk.roll({'kb1':-delta_i, 'kb2':-delta_j}).rename({'m':'n'})
            
            i = np.arange(self.nb1)[:, None]
            j = np.arange(self.nb2)[None, :]

            wi = np.where(i + delta_i < 0, -1, np.where(i + delta_i >= self.nb1, 1, 0))
            wj = np.where(j + delta_j < 0, -1, np.where(j + delta_j >= self.nb2, 1, 0))
            
            Gx = wi * self.b1[0] + wj * self.b2[0]
            Gy = wi * self.b1[1] + wj * self.b2[1]
            
            phase = np.exp(1j * (Gx[:,:,None,None] * u_mk.x.data[None,None,:,:] + Gy[:,:,None,None] * u_mk.y.data[None,None,:,:]))
            u_nkpb *= phase[None,:,:,:,:]
            
            M_mnkb.loc[{'b':b_idx}] = (u_mk.conjugate() * u_nkpb).sum(dim=['a1','a2'])
            
        return M_mnkb

    def r_n(self, M_mnkb: xr.DataArray) -> xr.DataArray:
        M_nnkb = M_mnkb.sel(m=M_mnkb.n)
        r = -(self.weights * self.neighbors * xr.ufuncs.angle(M_nnkb)).sum('b').sum(["kb1", "kb2"]) / self.nb1 / self.nb2
        return r
     
    def Omega(self, M_mnkb: xr.DataArray) -> xr.DataArray:        
        M_nnkb = M_mnkb.sel(m=M_mnkb.n)
        b_dot_rn = (self.neighbors * self.r_n(M_mnkb)).sum("kxy")
        diag = ((xr.ufuncs.angle(M_nnkb) + b_dot_rn) ** 2).sum("n")

        nondiag = (abs(M_mnkb) ** 2).where(M_mnkb.m != M_mnkb.n).sum(["m", "n"])

        omega = ((diag + nondiag)*self.weights).sum('b').sum(["kb1", "kb2"]) / self.nb1 / self.nb2

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
                {"kb1": -self.neighbors_indexes[ib, 0].item(), 
                 "kb2": -self.neighbors_indexes[ib, 1].item()}
            )
            nM[{"b": ib}] = matmul(
                trH(U), matmul(M[{"b": ib}], U_kpb, ["m", "n"]), ["m", "n"]
            )

        return nM

    def guess(self, u_mk:xr.DataArray, centers)->xr.DataArray:
        """Initialize a matrix U_mnk0 by using gaussian functions.

        Args:
            u_mk (xr.DataArray): The initial eigenvector set
            centers (_type_): The centers of each wannier functions as [x_list, y_list]

        Returns:
            xr.DataArray: _description_
        """
        # Taking random points for the centers
        
        sigma = self.potential.a1@self.potential.a1 / 10 # A reasonable spread
        
        g_n = xr.DataArray(
            np.zeros((self.n_wannier, u_mk.sizes['a1'], u_mk.sizes['a2'])),
            coords = {'n':np.arange(self.n_wannier), 'a1':u_mk.a1, 'a2':u_mk.a2}
        )
        
        for i in range(self.n_wannier):
            gauss = np.exp(
                - ((u_mk.x - centers[0][i])**2 + (u_mk.y - centers[1][i])**2) / 2 / (sigma*uniform(0.5, 2))**2
            )
            gauss /= (gauss**2).sum(["a1", "a2"])
            g_n[{'n':i}] = gauss
        
        # Now performing a lödwin decomposition
        A_mnk = (u_mk.conjugate() * g_n).sum(["a1", "a2"])
        S_mnk = matmul(trH(A_mnk), A_mnk, ["m", "n"])

        invsqrt_S_mnk = xr.apply_ufunc(
            lambda m: fractional_matrix_power(m, -1 / 2),
            S_mnk,
            input_core_dims=[["m", "n"]],
            output_core_dims=[["m", "n"]],
        )

        U_mnk0 = matmul(A_mnk, invsqrt_S_mnk, ["m", "n"]).transpose('m', 'n', 'kb1', 'kb2')
        
        return U_mnk0
                    
    def compute_bloch(self, n_wannier, **kwargs)->xr.DataArray:
        """Compute the Bloch eigenvectors necessary to the determination of the MLWFs.

        Args:
            n_wannier (_type_): The number of Wannier functions to compute.

        Returns:
            xr.DataArray: The bloch eigenvectors
        """
        
        self.n_wannier = n_wannier
        if self.method == 'pw':
            solv = PWSolver(
                self.potential, self.alpha, **kwargs
            )
            
            solv.set_reciprocal_space(self.kx, self.ky)
            eigva, eigve = solv.solve(n_wannier, parallel=True, n_cores = -1)
            eigve = solv.compute_u(eigve)
        
        elif self.method == 'fd':
            solv = FDSolver(self.potential, self.alpha)
            solv.set_reciprocal_space(self.kx, self.ky)
            eigva, eigve = solv.solve(n_wannier, parallel=True, n_cores = -1)

        else:
            raise ValueError("Method must either be 'pw' or 'fd'")
        
        if n_wannier == 1:
            eigve = eigve.expand_dims('band')
        
        self.eigve = eigve.transpose(... , 'band', 'kb1', 'kb2','a1','a2').rename({'band':'m'})

    
    def compute_U_mnk(self, sel:dict, centers:list[list[float]], tol:float)->xr.DataArray:
        """Determine the MLWFs for a given collection of eigenvectors u_mk at a given parameter space point 'sel',
        using a basic gradient descent algorithm.

        Args:
            sel (xr.DataArray): The point in parameter space for which to find the MLWFs.
            U_mnk0 (xr.DataArray): The initial guess for the Unitary matrix.

        Returns:
            xr.DataArray: The unitary matrix transformation U_mnk required to determine the MLWFs
        """
        u_mk = self.eigve.sel(sel)
        U_mnk0 = self.guess(u_mk, centers)
        
        M_init = self.M_mnkb(u_mk)
        U_init = U_mnk0
        
        M0 = self.new_M(U_init, M_init)
        U0 = U_init
        
        Omega0 = self.Omega(M0)  # Initial value of the functional
        n_up = 0
        alpha = 0.1

        while n_up < 10:
            G0 = self.G_nmk(M0)
            U_trial = matmul(U0, xexpm(alpha * G0), ["m", "n"])
            M_trial = self.new_M(U_trial, M_init)
            Omega_trial = self.Omega(M_trial)

            epsilon = Omega0 - Omega_trial
            if epsilon > tol:
                n_up = 0
                M0 = M_trial.copy()
                U0 = U_trial.copy()
                Omega0 = Omega_trial
                alpha *= 1.2
            else:
                n_up += 1
                alpha *= 0.5
            
            # print(Omega_trial)

        return U0
    
    def solve(
        self, 
        n_wannier: int,
        centers:list[list[float]],
        parallel: bool = False,
        n_cores: int = -1,
        blockwargs: dict = {}, 
        tol = 1e-7)->xr.DataArray:
        """Finds the proper unitary matrix U_mnk to compute the MLWFs at each point in parameter space.

        Args:
            n_wannier (int): The number of Bloch eigenvectors and thus WFs to compute.
            centers (list[list[float]]): The centers of the WFs, given as [[x0, ..., xN], [y0, ..., yN]].
            parallel (bool, optional): Wheter to parallelize the whole function. Defaults to False.
            n_cores (int, optional): Numbers of cores to use in case of parallelization. Defaults to -1.
            blockwargs (dict, optional): Arguments to pass on to the Bloch-Schrödinger solver constructor function. Defaults to {}.
            tol (_type_, optional): The tolreance for the gradient descent finding U_mnk. Defaults to 1e-7.

        Returns:
            xr.DataArray: The unitary matrix U_mnk with additional parameter dimensions.
        """
        print("Computing the Bloch functions...")
        self.compute_bloch(n_wannier, **blockwargs)
        
        paramcoords = {
            dim:self.eigve.coords[dim] for dim in self.eigve.dims
            if dim not in ['m', 'kb1', 'kb2', 'a1', 'a2']
        }
        
        allcoords = {
            **paramcoords,
            "m":self.eigve.m,
            "n":self.eigve.m.rename('n').rename({'m':'n'}),
            "kb1":self.eigve.kb1,
            "kb2":self.eigve.kb2,
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
                
        def f(x):
            return self.compute_U_mnk(x, centers, tol)
        
        print(f"Computing {n_tot} sets of Wannier functions")
        if parallel:
            parallel = Parallel(n_jobs=min(n_cores, n_tot), return_as="list", verbose = 5)
            results = parallel(delayed(f)(x) for x in selections)
        else:
            results = []
            with tqdm(total=n_tot) as pbar:
                for x in selections:
                    results += [f(x)]
                    pbar.update(1)

        for i in range(n_tot):
            U_tot_mnk.loc[selections[i]] = results[i]
        
        return U_tot_mnk

    def compute_wannier(
        self, 
        U_mnk:xr.DataArray,
        bounds1: tuple[int, int],
        bounds2: tuple[int, int],
        coarsen: tuple[int, int] = (1,1),
        )->tuple[Potential, xr.DataArray]:
        """Compute the WFs profiles from a given unitary matrix.

        Args:
            U_mnk (xr.DataArray): The unitary matrix to use, its dimensions must match those of the corresponding Bloch vectors.
            bounds1 (tuple[int, int]): The number of unit cells along a1 over which to extend the WFs computation.
            bounds2 (tuple[int, int]): The number of unit cells along a2 over which to extend the WFs computation.
            coarsen (tuple[int, int], optional): Wheter to coarsen the resolution of the mode profile. The coarsening factors must be dividers of na1 and na2. 
            See xarray coarsen function for more infos. Defaults to (1,1).

        Returns:
            tuple[Potential, xr.DataArray]: The extended potential for plotting/Hamiltonian computation as well as the MLWFs profiles.
        """
        if coarsen != (1,1):
            coarse_eig = self.eigve.coarsen(
                a1 = coarsen[0],
                a2 = coarsen[1],
                coord_func='min'
            ).mean()
        else:
            coarse_eig = self.eigve
            
        na1_coarse = self.potential.resolution[0] // coarsen[0]
        na2_coarse = self.potential.resolution[1] // coarsen[1]
        na1_tot = na1_coarse * (bounds1[1]-bounds1[0]) 
        na2_tot = na2_coarse * (bounds2[1]-bounds2[0])
        
        coords = {dim:coarse_eig.coords[dim] for dim in coarse_eig.dims if dim not in ['a1', 'a2', 'kb1', 'kb2']}

        coords.update(
            {'a1':np.linspace(bounds1[0]-1/2, bounds1[1]-1/2, na1_tot, endpoint=False) + 1/na1_coarse/2,
             'a2':np.linspace(bounds2[0]-1/2, bounds2[1]-1/2, na2_tot, endpoint=False) + 1/na2_coarse/2}
        )
        
        shape = tuple(
            [coord.shape[0] for coord in coords.values()]
        )
        
        wannier = xr.DataArray(
            np.zeros(shape, dtype = complex),
            coords=coords
        )
        
        tot_x = self.potential.a1[0] * wannier.a1 + self.potential.a2[0] * wannier.a2
        tot_y = self.potential.a1[1] * wannier.a1 + self.potential.a2[1] * wannier.a2
                
        wannier = wannier.assign_coords(
            {"x": tot_x, "y": tot_y}
        )
        
        print("Computing the mode profiles...")
        for ikb1 in trange(self.eigve.sizes['kb1']):
            for ikb2 in range(self.eigve.sizes['kb2']):
                for ia1 in range(bounds1[1]-bounds1[0]):
                    for ia2 in range(bounds2[1]-bounds2[0]):
                        
                        lcR = {
                            'a1':slice(ia1*na1_coarse, (ia1+1)*na1_coarse),
                            'a2':slice(ia2*na2_coarse, (ia2+1)*na2_coarse)
                        }
                        
                        
                        lcK = {
                            'kb1':ikb1,
                            'kb2':ikb2
                        }
                        
                        kx = self.kx[lcK].item()
                        ky = self.ky[lcK].item()
                        
                        x = wannier.x[lcR]
                        y = wannier.y[lcR]
                        
                        coarse_eig.coords['a1'] = x.a1
                        coarse_eig.coords['a2'] = x.a2
                        
                        phase = np.exp(-1j * (kx * x + ky * y))
                        tmp = 0
                        for i in range(U_mnk.sizes['m']):
                            tmp += (U_mnk[lcK] * coarse_eig[lcK])[{'m':i}]
                        tmp = tmp * phase
                        
                        # tmp = coarse_eig[lcK] * phase
                                                                    
                        wannier[lcR] += tmp.data

        wannier = wannier.rename({"m":"n"})
        wannier = wannier / (abs(wannier) ** 2).sum(["a1", "a2"]) ** 0.5
        tiled_pot = self.potential.coarsen(coarsen).tile(bounds1, bounds2)
        
        return tiled_pot, wannier
        
        

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt  # noqa: E402
    from bloch_schrodinger.plotting import plot_eigenvector, get_template  # noqa: E402

    
    kl = 2
    a = 4*np.pi / 3**1.5 / kl
    a1 = np.array([3 * a/2, -(3**0.5) * a/2])  # 1st lattice vector
    a2 = np.array([3 * a/2, 3**0.5 * a/2])  # 2nd lattice vector

    b1 = 3**0.5 * kl * np.array([1, -3**0.5])/2
    b2 = 3**0.5 * kl * np.array([1, 3**0.5])/2

    m = 1
    hbar = 1
    E_r = hbar**2 * kl**2 / 2 / m
    s1 = 9
    s2 = 10

    k1 = kl * np.array([-3**0.5 / 2 , 1/2])
    k2 = kl * np.array([3**0.5 / 2 , 1/2])
    k3 = kl * np.array([0,-1])

    a1s = np.array([-1, 3**0.5])*2*np.pi/3/a 
    a2s = np.array([1, 3**0.5])*2*np.pi/3/a
    K = np.array([0, 4*np.pi/3**1.5/a])


    klim = 1
    na1 = 64
    na2 = 64
    
    centers = [
        [-a/2, a/2], [0, 0]
    ]

    V1 = -s2*E_r
    V2 = -s1*E_r

    honeycomb = Potential(
        unitvecs = [a1, a2],
        resolution = (na1, na2),
        v0 = 0,
    )

    dirs = [
        k1[0] * (honeycomb.x - a1[0]) + k1[1] * honeycomb.y,
        k2[0] * (honeycomb.x - a1[0]) + k2[1] * honeycomb.y,
        k3[0] * (honeycomb.x - a1[0]) + k3[1] * honeycomb.y,
    ]

    for i in range(3):
        honeycomb.add(2*V1*np.cos((dirs[i-1]-dirs[i]) - 2*np.pi/3)/2)
        honeycomb.add(2*V2*np.cos((dirs[i-1]-dirs[i]) + 2*np.pi/3)/2)

    foo = Wannier(
        Potential=honeycomb,
        alpha = 1/2,
        rec_vecs=[a1s, a2s],
        resolution=(15,15),
        method='pw'
    )
    
    
    U_mnk = foo.solve(2, centers, parallel=False, n_cores=2)    

    pot, wannier = foo.compute_wannier(
        U_mnk,
        [-1, 2],
        [-1, 2],
        coarsen=(2,2)
    )
    
    #%%

    amplog = get_template('amplitude - log')
    # amplog['contourkwargs']['levels'] = [10]
    amplog['autoscale'] = False
    amplog['clim'] = (1e-12, 1e-1)

    reallog = get_template('real')
    # reallog['contourkwargs']['levels'] = [1]
    # reallog['autoscale'] = False
    # reallog['clim'] = (-1e-1, 1e-1)
    
    fig, ax = plot_eigenvector(
        [[abs(wannier)**2]], [[pot]], [[amplog]]
    )
    plt.show()
    
    
    # bar = PWSolver(
    #     honeycomb, 1/2, 300
    # )
    
    # eigva, eigve = bar.solve(2)
    
    # eigve = bar.compute_u(eigve)
    
    # fig, ax = plot_eigenvector(
    #     [[abs(eigve)**2]], [[honeycomb]], [['amplitude']]
    # )
    # plt.show()
    

# %%
