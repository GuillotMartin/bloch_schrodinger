import numpy as np
import xarray as xr
from typing import Union
from bloch_schrodinger.potential import Potential, honeycomb
import scipy.sparse as sps
from tqdm import tqdm
from scipy.sparse.linalg import lobpcg, LinearOperator, spilu, eigsh
import warnings

def real(arr:xr.DataArray)->xr.DataArray:
    return xr.apply_ufunc(np.real, arr)

def imag(arr:xr.DataArray)->xr.DataArray:
    return xr.apply_ufunc(np.imag, arr)


def check_name(name:str):
    """Check wether the name is a valid one. and raises an error if not.

    Args:
        name (str): The name to check

    Raises:
        ValueError: If the name is forbidden
    """
    
    forbidden_names = ['field', 'band', 'a1', 'a2', 'x', 'y', 'dx', 'dy', 'dx2', 'dy2', 'lap']
    if name in forbidden_names:
        raise ValueError(f"{name} is not a valid name for the object, as it is already used. The forbidden names are: {forbidden_names}")

class Solver:
    """The Solver class handles the heavy lifting of this package, constructing the Hamiltonian matrix and performing its efficient diagonalization
    """
    
    def __init__(self, potentials:Union[Potential,list[Potential]],alphas:Union[Union[float,xr.DataArray],list[Union[float,xr.DataArray]]]):
        """Instantiate the solver.

        Args:
            potentials (Union[Potential,list[Potential]]): The potential felt by each field. They must all be defined on the same grid. 
            A single potential can also be passed for a scalar equation.
            alphas (Union[Potential,list[Potential]]): The kinetic energy coefficient hbar²/2m for each field. 
            A single coefficient can be passed for a scalar equation

        Raises:
            ValueError: Not the same number of potentials and kinetic terms given.
        """
        
        if isinstance(potentials, Potential) and (isinstance(alphas, float) or isinstance(alphas, int) or isinstance(alphas, xr.DataArray)):
            self.potentials = [potentials]
            self.alphas = [alphas]
            self.nb = 1
        else: 
            self.potentials = potentials
            self.alphas = alphas
            if len(potentials) == len(alphas):self.nb = len(potentials)
            else:raise ValueError("potentials and alphas not of the same length")
        
        
        # storing all parameter coordinates from potentials and alphas. The final solver will run on all these dimensions.
        self.allcoords = {}
        for pot in self.potentials:
            coords_pot = {dim:['potential',pot.V.coords[dim]] for dim in pot.V.dims if dim not in ['a1','a2']}
            self.allcoords.update(coords_pot)
        
        for alph in self.alphas:
            if isinstance(alph, xr.DataArray):
                for dim in alph.dims:
                    check_name(dim)
                coords_alpha = {dim:['alpha',alph.coords[dim]] for dim in alph.dims}
                self.allcoords.update(coords_alpha)
        
        self.a1 = self.potentials[0].a1 # The first lattice vector
        self.a2 = self.potentials[0].a2 # The second lattice vector
        self.e1 = self.a1/(self.a1@self.a1)**0.5 # normalized lattice vector
        self.e2 = self.a2/(self.a2@self.a2)**0.5 # normalized lattice vector
        self.na1 = len(self.potentials[0].V.a1.data) # discretization along a1
        self.na2 = len(self.potentials[0].V.a2.data) # discretization along a2
        self.np = self.na1*self.na2 # Number of mesh sampling points
        
        self.n = self.np * self.nb # Total number of points, when counting all the fields
                        
        self.g = np.linalg.inv(np.array( # The metric, important to compute derivative operators
            [[1,self.e1@self.e2],
            [self.e2@self.e1,1]]
        )) 
        
        # length steps along a1 and a2
        self.da1 = float(abs(self.potentials[0].V.a1[1]-self.potentials[0].V.a1[0]))*(self.a1@self.a1)**0.5 # smallest increment of length along a1
        self.da2 = float(abs(self.potentials[0].V.a2[1]-self.potentials[0].V.a2[0]))*(self.a2@self.a2)**0.5 # smallest increment of length along a2
        
        # Initializing the coupling evalution context and the coupling expressions
        self.coupling_context = {
            "np":np,
            "real":real,
            "imag":imag
        }
        self.couplings = []
        
        # computing the sparse matrices needed
        self.potential_matrix()
        self.compute_d1()
        self.compute_d2()
        self.compute_grad()
        self.compute_laplacian()
        
    def __repr__(self)->str:
        shape = {dim:len(self.allcoords[dim][1].data) for dim in self.allcoords}
        return f"Solver object \n size ({self.n}, {self.n}), with {self.nb} field(s) \n dimensions: {shape}"

    def potential_matrix(self):
        """prepare the potentials in a shape that can be instanciated as sparse matrices very efficiently
        """
    
        # flattening the potentials
        Vflats = [pot.V.stack(idiag=['a1','a2']) for pot in self.potentials]
        
        # Expanding each potential dimensions to the full parameter space, composed of all dimensions of each potential
        for i, V in enumerate(Vflats):
            V = V.drop_vars(['idiag', 'a1', 'a2']).assign_coords(idiag=('idiag', np.arange(i*self.np, (i+1)*self.np)))
            for d, coords in self.allcoords.items():
                if coords[0] == 'potential':
                    if d not in V.dims:
                        V = V.expand_dims({d: coords[1]})
            Vflats[i] = V
        
        # data in the shape (...,self.n) with the first dimensions being all the parameter dimensions of potentials
        self.potential_data = xr.concat(Vflats,dim='idiag')
    
    def compute_d2(self):
        """Compute the derivatation operator along the direction a1
        """
        diag1 = np.ones((self.np-1), dtype = int)
        diag1[self.na2-1::self.na2] = 0
        
        diag2 = np.zeros((self.np-self.na2+1), dtype = int)
        diag2[0::self.na2] = 1
        
        diag0 = np.ones(self.np)*2
        
        diags = [
            np.append(diag2, np.zeros(self.na2-1)),
            -np.append(diag1, np.zeros(1)), 
            np.insert(diag1, 0, np.zeros(1)), 
            -np.insert(diag2, 0, np.zeros(self.na2-1)),
        ]
        offs = [-self.na2+1,-1,1,self.na2-1]
                
        self.d2 = sps.dia_array((diags, offs), dtype = float, shape = (self.np,self.np))/self.da2/2

        diags_sec = [
            np.append(diag2, np.zeros(self.na2-1)),
            np.append(diag1, np.zeros(1)),
            -diag0,
            np.insert(diag1, 0, np.zeros(1)), 
            np.insert(diag2, 0, np.zeros(self.na2-1)),
        ]
        offs_sec = [-self.na2+1, -1, 0, 1, self.na2-1]
                
        self.d2_sec = sps.dia_array((diags_sec, offs_sec), dtype = float, shape = (self.np,self.np))/self.da2**2

    def compute_d1(self):
        """Compute the derivatation operator along the direction a1
        """
        diag1 = np.ones((self.na2), dtype = int)
        diag0 = np.ones(self.np)*2
        diag2 = np.ones((self.np-self.na2), dtype = int)
        
        diags = [
            np.append(diag1, np.zeros(self.np-self.na2)),
            -np.append(diag2, np.zeros(self.na2)), 
            np.insert(diag2, 0, np.zeros(self.na2)), 
            -np.insert(diag1, 0, np.zeros(self.np-self.na2)),
        ]
        
        offs = [-self.np+self.na2,-self.na2,self.na2,self.np-self.na2]
                
        self.d1 = sps.dia_array((diags, offs), dtype = float, shape = (self.np,self.np))/self.da1/2

        diags_sec = [
            np.append(diag1, np.zeros(self.np-self.na2)),
            np.append(diag2, np.zeros(self.na2)),
            -diag0,
            np.insert(diag2, 0, np.zeros(self.na2)), 
            np.insert(diag1, 0, np.zeros(self.np-self.na2)),
        ]
        offs_sec = [-self.np+self.na2, -self.na2 ,0, self.na2, self.np-self.na2]
                
        self.d1_sec = sps.dia_array((diags_sec, offs_sec), dtype = float, shape = (self.np,self.np))/self.da1**2

    def compute_grad(self):
        """compute the gradient and derivation operators"""
        # We use the generalized coordinate formula for the gradient
        self.grad = [
            (self.e1[k]*self.g[0,0] + self.e2[k]*self.g[0,1])*self.d1 +
            (self.e1[k]*self.g[1,0] + self.e2[k]*self.g[1,1])*self.d2
            for k in range(2)
        ]
        self.dx, self.dy = self.grad

    def compute_laplacian(self):
        """Compute the laplacian and second order derivation operators """
        self.lap = (
            self.g[0,0]*self.d1_sec+
            self.g[0,1]*self.d2@self.d1+
            self.g[1,0]*self.d1@self.d2+
            self.g[1,1]*self.d2_sec            
        )
        
        # shorter notations
        g = self.g
        e = [self.e1, self.e2]
        # The Hessian defines the second order derivative in the e1,e2 basis
        Hess = [[self.d1_sec, self.d1 @ self.d2],
                [self.d2 @ self.d1, self.d2_sec]]
        
        self.dx_sec = sps.dia_array((self.np, self.np))
        self.dy_sec = sps.dia_array((self.np, self.np))
        self.dxdy   = sps.dia_array((self.np, self.np))
        self.dydx   = sps.dia_array((self.np, self.np))
                
        for mu in range(2):
            for nu in range(2):
                for i in range(2):
                    for j in range(2):
                        self.dx_sec += Hess[mu][i] * g[i,j] * g[mu,nu] * e[j][0] * e[nu][0]
                        self.dy_sec += Hess[mu][i] * g[i,j] * g[mu,nu] * e[j][1] * e[nu][1]
                        self.dxdy   += Hess[mu][i] * g[i,j] * g[mu,nu] * e[j][0] * e[nu][1]
                        self.dydx   += Hess[mu][i] * g[i,j] * g[mu,nu] * e[j][1] * e[nu][0]
                                 
    def compute_full_operators(self, alphas:list[float]):
        """Compute all the total kinetic operators for each fields, as bloch sparse matrices
        
        Args:
            alphas (list[float]): The list of alphas to use
        """
        
        # Initialization as lists of blocks
        alphaIdent = [alphas[u]*sps.eye_array(n=self.np,m=self.np) for u in range(self.nb)]
        alphaDx = [alphas[u]*self.dx for u in range(self.nb)]
        alphaDy = [alphas[u]*self.dy for u in range(self.nb)]
        alphaLap = [alphas[u]*self.lap for u in range(self.nb)]
        # create the full matrices
        self.alphaIdent = sps.block_diag(alphaIdent)
        self.alphaDx = sps.block_diag(alphaDx)
        self.alphaDy = sps.block_diag(alphaDy)
        self.alphaLap = sps.block_diag(alphaLap)
            
    def compute_kinetic(self, k:tuple[float] = (0,0))->sps.dia_matrix:
        """Compute the total kinetic operator.

        Args:
            k (tuple[float]): The k vector. default to (0,0)
        """
        return -self.alphaLap + 2*1j*(k[0]*self.alphaDx + k[1]*self.alphaDy) + (k[0]**2+k[1]**2)*self.alphaIdent
        
    def set_reciprocal_space(self, kx:Union[float,xr.DataArray], ky:Union[float,xr.DataArray]):
        """Add the reciprocal space to the list of coordinates.

        Args:
            kx (Union[float,xr.DataArray]): kx coordinate, can be a single float, a 1D xarray coordinate or even a multidimensional coordinate.
            ky (Union[float,xr.DataArray]): ky coordinate, can be a single float, a 1D xarray coordinate or even a multidimensional coordinate.
        """
        self.allcoords.update({'kx':['reciprocal',kx]})
        self.allcoords.update({'ky':['reciprocal',ky]})
        
    def create_reciprocal_grid(self, kx:Union[float,np.ndarray] = 0, ky:Union[float,np.ndarray] = 0):
        """Create the k-space grid on which the eigenvalues and vectors will be computed

        Args:
            kx (float or np.ndarray): The values of kx for the grid points. default to 0.
            ky (float or np.ndarray): The values of ky for the grid points. default to 0.
        """
        if isinstance(kx, float) or isinstance(kx, int):
            kx = xr.DataArray(np.array([kx]), coords = {'kx':np.array([kx])}, dims = 'kx')
        else:
            kx = xr.DataArray(kx, coords = {'kx':kx}, dims = 'kx')
            
        if isinstance(ky, float) or isinstance(ky, int):
            ky = xr.DataArray(np.array([ky]), coords = {'ky':np.array([ky])}, dims = 'ky')
        else:
            ky = xr.DataArray(ky, coords = {'ky':ky}, dims = 'ky')
            
        self.set_reciprocal_space(kx,ky)
    
    def add_coupling_parameter(self, name:str, parameter:Union[float,int,complex,np.ndarray,xr.DataArray]):
        """Add a parameter to the coupling evaluation context or to the list of dimensions.

        Args:
            name (str): The name of the coupling parameter, if the parameter given is an xarray, its name will be overwritten by this argument. must be unique.
            parameter (Union[float,int,complex,xr.DataArray]): The value(s) of the parameter.
        """
        check_name(name)
        
        if name in self.coupling_context:
            raise ValueError(f"{name} already present in the coupling context")
        
        if name in self.coupling_context:
            raise ValueError(f"{name} already present in the coupling context")
        
        if type(parameter) == float or type(parameter) == complex or type(parameter) == int:
            self.coupling_context.update({name:parameter})
        else:
            if isinstance(parameter,np.ndarray):
                parameter = xr.DataArray(
                    data = parameter,
                    coords = {name:parameter},
                    name = name
                )
            elif isinstance(parameter,xr.DataArray):
                parameter.rename(name)
            else:
                raise TypeError("Type of parameter not supported")
            
            self.allcoords.update({name:['coupling',parameter]})
    
    def add_coupling_matrix(self, name:str, struct:sps.spmatrix, field1:int, field2:int):
        """Add a sparse matrix to the coupling context, to be evaluated during Hamiltonian construction.

        Args:
            name (str): The name of the matrix to call during evaluation, must be unique.
            struct (sps.spmatrix): A (self.np,self.np) sparse matrix containing the structure of the coupling term.
            hermitian (bool): Wheter to add also the  
        """
        
        check_name(name)
        
        shape = [[None,None],
                 [None,None]]
        shape[field1][field2] = struct
        shape[field2][field1] = struct.transpose().conj()

        matrix = sps.block_array(shape)
        self.coupling_context.update({name:matrix})
        
    def add_coupling(self, expr:str):
        """Add a coupling expression to the list of couplings. The coupling expression is a string that will be evaluated at Hamiltonian building phase. 
        It can use all parameters and matrices set by 'add_coupling_parameter' and 'add_coupling_matrices', as well as the other parameters of the potentials and alphas.
        Be careful to not multiply a matrix by an imaginary number, as it will break hermiticity. You should instead create a purely imaginary Hermitian matrix then multiply it by a real value

        Args:
            expr (str): The string to evaluate.
        """
        self.couplings += [expr]
    
    def add_on_site_coupling(self, expr:str, field1:int, field2:int):
        """Add a simple, local coupling term between two field. The complex conjugate term is applied automatically.

        Args:
            name (str): The name of the coupling, must be unique.
            expr (str): An expression describing the stength of the coupling, must only use parameters that are in (or will be added to) the context manager.
            field1 (int): The origin field of the coupling.
            field2 (int): The target field of the coupling.
        """

        matrix = sps.eye(self.np, self.np)
        mat_name = f"id_{field1}{field2}"
        imat_name = f"i_id_{field1}{field2}"
        self.add_coupling_matrix(mat_name, matrix, field1, field2)
        self.add_coupling_matrix(mat_name, 1j*matrix, field1, field2)
        
        self.add_coupling(f"real({expr}) * {mat_name} + imag({expr}) * {imat_name}")
        
    def add_TETM(self, expr:str, field1:int, field2:int):
        """Add a TETM splitting term between fields 1 and 2

        Args:
            expr (str): An expression describing the stength of the coupling, must only use parameters that are in (or will be added to) the context manager.
            field1 (int): The origin field of the coupling.
            field2 (int): The target field of the coupling.
        """
        f1, f2 = field1, field2 #shorten names for expression length
        
        # adding the real and imaginary main diagonal
        matrix = sps.eye(self.np, self.np)
        mat_name = f"id_{f1}{f2}"
        self.add_coupling_matrix(mat_name, matrix, field1, field2)
        
        mat_name = f"i_id_{f1}{f2}"
        self.add_coupling_matrix(mat_name, 1j*matrix, field1, field2)

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
        self.coupling_context.update({'kx':0, 'ky':0})
        
        # The expression of the TE/TM splitting operator
        expr_coup = (
            f"(ky**2 - kx**2)*id_{f1}{f2} + 2*kx*ky*i_id_{f1}{f2} + " + # zeroth-order term
            f"2 * (kx * (idx_{f1}{f2} + dy_{f1}{f2}) + ky * (dx_{f1}{f2} - idy_{f1}{f2})) + " + # first order term
            f"dx_sec_{f1}{f2} - dy_sec_{f1}{f2} - idxdy_{f1}{f2} - idydx_{f1}{f2}" # second order term
        )

        
        self.add_coupling(f"{expr} * ({expr_coup})")
    
    def initialize_eigva(self, n_eigva:int)->xr.DataArray:
        """Initialize the array containing the eigenvalues

        Args:
            n_eigva (int): The number of eigenvalues to compute

        Returns:
            xr.DataArray: An empty DataArray with the proper shape.
        """
        eigva_coords = [coord[1] for dim, coord in self.allcoords.items()]
        
        eigva_coords += [xr.DataArray(
            np.arange(n_eigva), 
            coords = {'band':np.arange(n_eigva)}
        )]
        
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
        shape = [all_coords[d].sizes[d] if isinstance(all_coords[d], xr.DataArray) 
                else len(all_coords[d]) 
                for d in all_coords]

        # 3. Create the zero-filled DataArray
        eigva = np.zeros(shape)
        eigva = xr.DataArray(eigva, dims=list(all_coords.keys()), coords=all_coords, name='eigva')
        
        return eigva
    
    def initialize_eigve(self, n_eigva:int)->xr.DataArray:
        """Initialize the array containing the flattened eigenvectors

        Args:
            n_eigva (int): The number of eigenvalues to compute

        Returns:
            xr.DataArray: An empty DataArray with the proper shape.
        """
        eigve_coords = [coord[1] for dim, coord in self.allcoords.items()]
        
        
        eigve_coords += [xr.DataArray(
            np.arange(self.nb), 
            coords = {'field':np.arange(self.nb)}
        )]

        eigve_coords += [xr.DataArray(
            np.linspace(0,1,self.na1), 
            coords = {'a1':np.linspace(0,1,self.na1)}
        )]
        
        eigve_coords += [xr.DataArray(
            np.linspace(0,1,self.na2), 
            coords = {'a2':np.linspace(0,1,self.na2)}
        )]        

        eigve_coords += [xr.DataArray(
            np.arange(n_eigva), 
            coords = {'band':np.arange(n_eigva)}
        )]

        
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
        shape = [all_coords[d].sizes[d] if isinstance(all_coords[d], xr.DataArray) 
                else len(all_coords[d]) 
                for d in all_coords]

        # 3. Create the zero-filled DataArray
        eigve = np.zeros(shape, dtype = complex)
        eigve = xr.DataArray(eigve, dims=list(all_coords.keys()), coords=all_coords, name='eigve_flat')
        
        x = self.a1[0]*eigve.a1 + self.a2[0]*eigve.a2
        y = self.a1[1]*eigve.a1 + self.a2[1]*eigve.a2
        eigve = eigve.assign_coords({
            "x":x,
            "y":y,
        })
        
        eigve = eigve.stack(component = ('field','a1','a2')).transpose(..., 'component','band')
        return eigve

    def make_alpha_list(self, sel:dict)->list[float]:
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
            if isinstance(self.alphas[u],float) or isinstance(self.alphas[u],int):
                alphas += [self.alphas[u]]
            elif isinstance(self.alphas[u],xr.DataArray):
                sub_sel = {dim:value for dim,value in sel.items() if dim in self.alphas[u].dims}
                alphas += [float(self.alphas[u].sel(sub_sel).data)]
            else:
                raise TypeError(f"{u}-th alpha term not of a recognized type (int, float or xr.DataArray)")
        return alphas

    def solve(self, n_eigva:int, keep_vecs:bool = 'True')->tuple[xr.DataArray]:
        """Solve the eigenvalue problem at every points of the parameter space, using the scipy.sparse.eigsh function. This function is painfully parallelizable, 
        you are welcome to create your own based on your architecture and needs. The eigenvalues and vectors are returned as properly shaped DataArrays.

        Args:
            n_eigva (int): The number of eigenvalues to compute.
            keep_vecs (bool, optional): Wether to keep the eigenvectors. Defaults to 'True'.
        Returns:
            tuple[xr.DataArray]: the eigenvalues and the eigenvectors if keep_vecs is True.
        """

        # Create empty DataArrays to store the eigenvalues and vectors
        eigva = self.initialize_eigva(n_eigva)
        if keep_vecs:
            eigve = self.initialize_eigve(n_eigva)

        # Create lists of dimensions. The dimensions are separated by type, for efficient Hamiltonian matrix construction.
        potential_dims = [dim for dim in self.allcoords if self.allcoords[dim][0] == 'potential']
        reciprocal_dims = [dim for dim in self.allcoords if self.allcoords[dim][0] == 'reciprocal']
        alphas_dims = [dim for dim in self.allcoords if self.allcoords[dim][0] == 'alpha']
        couplings_dims = [dim for dim in self.allcoords if self.allcoords[dim][0] == 'coupling']
        
        # Intializing indexes to loop over.
        potential_index, alpha_index, reciprocal_index, coupling_index = [()], [()], [(0,0)], [()]
        
        # Stacking the dimensions of each type and replacing the associated index. If there is no dimensions of a given type,
        # the corresponding index is kept as the default defined above
        if len(potential_dims)>0:
            eigva = eigva.stack(potdims = potential_dims)
            potential_index = eigva.potdims.to_index()
            if keep_vecs:
                eigve = eigve.stack(potdims = potential_dims)
        if len(alphas_dims)>0:
            eigva = eigva.stack(alphadims = alphas_dims)
            alpha_index = eigva.alphadims.to_index()
            if keep_vecs:
                eigve = eigve.stack(alphadims = alphas_dims)
        if len(reciprocal_dims)>0:
            eigva = eigva.stack(recdims = reciprocal_dims)
            reciprocal_index = eigva.recdims.to_index()
            if keep_vecs:
                eigve = eigve.stack(recdims = reciprocal_dims)
        if len(couplings_dims)>0:
            eigva = eigva.stack(coupdims = couplings_dims)
            coupling_index = eigva.coupdims.to_index()
            if keep_vecs:
                eigve = eigve.stack(coupdims = couplings_dims)


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
            potential_sel = dict(zip(potential_index.names, pots)) if hasattr(potential_index, "names") else {}
            
            # The potential is a diagonal matrix, which we stored as a data array.
            potdiag = self.potential_data.sel(potential_sel).data
            potential_matrix = sps.diags(potdiag, offsets=0)
            
            for alphs in alpha_index:
                # Same for the alphas, using the 'make_alpha_list' class method
                alpha_sel = dict(zip(alpha_index.names, alphs)) if hasattr(alpha_index, "names") else {}
                alphas = self.make_alpha_list(alpha_sel)
                self.compute_full_operators(alphas)
                
                for recs in reciprocal_index:
                    reciprocal_sel = dict(zip(reciprocal_index.names, recs)) if hasattr(reciprocal_index, "names") else {}
                    kinetic_matrix = self.compute_kinetic(recs)
                    for coups in coupling_index:
                        coupling_sel = dict(zip(coupling_index.names, coups)) if hasattr(coupling_index, "names") else {}
                        total_sel = {**potential_sel,**alpha_sel,**reciprocal_sel, **coupling_sel} # aggregating all the values of each selections
                        ham = kinetic_matrix + potential_matrix # The initial Hamiltonian contains only the kinetic operator and the potential operator 
                        
                        # --- Add the coupling terms to the Hamiltonian ---
                        self.coupling_context.update(total_sel)
                        for coupling in self.couplings:
                            ham += eval(coupling, {"__builtins__": {}}, self.coupling_context)
      
                        
                        with warnings.catch_warnings(record=True) as caught:
                            # "always" makes them get appended to caught so they are not printed
                            warnings.simplefilter("always")
                            eigvals, eigvecs = eigsh(ham, k = n_eigva, v0 = X[:,0], which = 'SM')                        
                        idx = eigvals.argsort()
                        eigvals = eigvals[idx]
                        eigvecs = eigvecs[:, idx]
                        
                        X = eigvecs
                        
                        eigva.loc[total_sel] = eigvals
                        if keep_vecs:
                            eigve.loc[total_sel] = eigvecs
                        pbar.update(1)
                        count +=1
        pbar.close()
        
        if len(potential_dims)>0:
            eigva = eigva.unstack(dim='potdims')
            if keep_vecs:
                eigve = eigve.unstack(dim='potdims')
        if len(alphas_dims)>0:
            eigva = eigva.unstack(dim='alphadims')
            if keep_vecs:
                eigve = eigve.unstack(dim='alphadims')
        if len(reciprocal_dims)>0:
            eigva = eigva.unstack(dim='recdims')
            if keep_vecs:
                eigve = eigve.unstack(dim='recdims')
        if len(couplings_dims)>0:
            eigva = eigva.unstack(dim='coupdims')
            if keep_vecs:
                eigve = eigve.unstack(dim='coupdims')


        
        if keep_vecs:
            eigve = eigve.unstack(dim='component').rename('eigve')
            eigve = eigve * xr.ufuncs.exp(1j*xr.ufuncs.angle(eigve.sel(a1=0.5,a2=0.55, field = 0, method='nearest')))
            x = self.a1[0]*eigve.a1 + self.a2[0]*eigve.a2
            y = self.a1[1]*eigve.a1 + self.a2[1]*eigve.a2
            eigve = eigve.assign_coords({
                "x":x,
                "y":y,
            })
            
        if keep_vecs:
            return eigva.squeeze(), eigve.squeeze()
        else:
            return eigva.squeeze()
    

import matplotlib.pyplot as plt
if __name__ == '__main__':
    
    res = (6,6)
    P = Potential([[3,-2], [3,2]], resolution=res,v0 = 0)
    P2 = Potential([[2,0], [0,2]], resolution=res,v0 = 0)
    
    P.V = P.V.x**2 - P.y**2
    
    solv = Solver([P,P2], [1,1])
    solv.add_TETM('1', 0, 1)
    op = eval(solv.couplings[0], {"__builtins__": {}}, solv.coupling_context)#[:res[0]**2, res[0]**2:]
    # op = solv.lap
    # op = solv.coupling_context['dx_sec_01']
    Vflat = P.V.data.reshape(-1)
    
    
    plt.imshow(np.imag(op.toarray()), vmin = -1, vmax = 1, cmap = 'coolwarm')
    plt.show()
    
    # tpflat = op @ Vflat
    # tp = tpflat.reshape(res)
    
    # fig, ax = plt.subplots()
    # im = ax.pcolormesh(P.x[1:-1,1:-1], P.y[1:-1,1:-1], np.imag(tp)[1:-1,1:-1])
    # plt.colorbar(im)
    # ax.set_aspect('equal')
    # plt.show()