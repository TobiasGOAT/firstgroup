import numpy as np
import scipy.sparse as sp

"""
------------------------------------------------------------
READ ME FIRST 
------------------------------------------------------------
A finite-difference solver for the 2D Laplace equation
on a rectangular room.

It builds a sparse matrix with the 5-point stencil,
applies boundary conditions, and solves Au = b for the temperature.

Coordinate picture:
y=Ly  ┌──────── top ────────┐
      │                     │
      │         Ω           │
left  │                     │  right
x=0   │                     │  x=Lx
      │                     │
      └────── bottom ───────┘  y=0

------------------------------------------------------------
Parameters
------------------------------------------------------------
dx : float
    Grid spacing is assumed to be the same in x and y axis.

sides : [Lx, Ly]
    Physical lengths of the rectangle. The grid sizes are:
        Nx = int(Lx/dx) + 1
        Ny = int(Ly/dx) + 1
    The 1D flattened solution has length N = Nx * Ny.

dirichletBC : [bottom, left, top, right]
neumanBC    : [bottom, left, top, right]
    Dirichlet values (temperatures) and Neumann fluxes (∂u/∂n) per side.
    For each side, provide EITHER Dirichlet OR Neumann (not both).

    Array lengths must match the number of nodes on each side:
        - bottom/top length: Nx
        - left/right length: Ny
    Use None for a side you don't set for that BC type.

-----------------------------------------------------------
Returns from solve(dirElseNeu: bool)
------------------------------------------------------------
u : (Nx*Ny,) ndarray
    Flattened temperature field (row-major; index = j*Nx + i).

sideValues : list of 1D ndarrays (length 4, order: [bottom, left, top, right])
    If dirElseNeu=True  → Dirichlet traces u on each side.
    If dirElseNeu=False → Neumann traces (∂u/∂n) estimated by
                           (u_inner - u_boundary)/dx  (aligned with outward normal).
    Lengths are Nx for bottom/top and Ny for left/right.
"""

class HeatSolver:
    def __init__(self, dx, sides, dirichletBC, neumanBC):
        """
        sides      : [Lx, Ly]
        dirichletBC: [bottom, left, top, right]
        neumanBC   : [bottom, left, top, right]
        """
        #in case the user gives too small grid
        self.dx = float(dx)
        Lx, Ly = float(sides[0]), float(sides[1])
        self.N_x = int(Lx / self.dx) + 1
        self.N_y = int(Ly / self.dx) + 1
        self.dirichletBC = dirichletBC
        self.neumanBC = neumanBC

        Nx, Ny = self.N_x, self.N_y
        if Nx < 2 or Ny < 2:
            raise ValueError("You messed up, the grid too small: need N_x >= 2 and N_y >= 2.")

        #Validate boundary-condition inputs
        def _check_side(name, dir_arr, neu_arr, expected_len):
            if dir_arr is not None and neu_arr is not None:
                raise ValueError(f"{name}: cannot set both Dirichlet and Neumann.")
            if dir_arr is not None and len(dir_arr) != expected_len:
                raise ValueError(f"{name} Dirichlet length {len(dir_arr)} != expected {expected_len}.")
            if neu_arr is not None and len(neu_arr) != expected_len:
                raise ValueError(f"{name} Neumann length {len(neu_arr)} != expected {expected_len}.")

        _check_side("bottom", dirichletBC[0], neumanBC[0], Nx)
        _check_side("left",   dirichletBC[1], neumanBC[1], Ny)
        _check_side("top",    dirichletBC[2], neumanBC[2], Nx)
        _check_side("right",  dirichletBC[3], neumanBC[3], Ny)

        #Build Laplacian matrix A (scaled), and boundary index ranges
        self.K_size = Nx * Ny
        self._createRanges()                  #boundary & adjacent-inner indices

        self.K,self.b = self._construct(dirichletBC,neumanBC)                 #unscaled 5-point stencil
 
        self.K = self.K.tocsr()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _construct(self,dirichletBC,neumanBC):
        """Unscaled 5-point Laplacian in LIL format (row-major flattening)."""
        main = -4.0 * np.ones(self.K_size)
        lr = np.ones(self.K_size - 1)
        # prevent wrap-around between rows for left/right neighbors
        lr[np.arange(1, self.N_y) * self.N_x - 1] = 0.0
        ud = np.ones(self.K_size - self.N_x)
        K = sp.diags(
            diagonals=[main, lr, lr, ud, ud],
            offsets=[0, 1, -1, self.N_x, -self.N_x],
            format='lil'
        )

        K = K / (self.dx ** 2)      #scale by 1/h^2 (before BCs)

        b = np.zeros((self.K_size,), dtype=float)   #Build RHS and apply boundary conditions

        # Apply BCs per side in order: [bottom, left, top, right]
        self._applyBC(K,b,dirichletBC[0], neumanBC[0], self.ranges[0][0])   #bottom
        self._applyBC(K,b,dirichletBC[1], neumanBC[1], self.ranges[0][1])   #left
        self._applyBC(K,b,dirichletBC[2], neumanBC[2], self.ranges[0][2])   #top
        self._applyBC(K,b,dirichletBC[3], neumanBC[3], self.ranges[0][3])   #right

        return K,b

    def _createRanges(self):
        Nx, Ny = self.N_x, self.N_y
        N = self.K_size

        #boundary indices
        bottom = np.arange(0, Nx)
        left   = np.arange(0, N, Nx)
        top    = np.arange(N - Nx, N)
        right  = np.arange(Nx - 1, N, Nx)

        #adjacent inner indices (one step inward)
        bottom_in = bottom + Nx
        left_in   = left + 1
        top_in    = top - Nx
        right_in  = right - 1

        self.ranges = [
            [bottom, left, top, right],
            [bottom_in, left_in, top_in, right_in]
        ]
        self.range_map = {
            "bottom": (bottom, bottom_in),
            "left":   (left,   left_in),
            "top":    (top,    top_in),
            "right":  (right,  right_in),
        }

    def _applyBC(self, K, b, DBCList, NBCList, boundary_indices):
        #Neumann first (only if provided)
        if NBCList is not None:
            for i, g in zip(boundary_indices, NBCList):
                K[i, i] += 1.0 / (self.dx ** 2)
                b[i]    += -float(g) / self.dx

        #Dirichlet overwrites (only if provided)
        if DBCList is not None:
            for i, val in zip(boundary_indices, DBCList):
                K.rows[i] = [i]         #efficient in LIL: set row to single entry
                K.data[i] = [1.0]
                b[i]      = float(val)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def updateBC(self,dirichletBC,neumanBC):
        #If DBC or NBC should not be uppdated, set them as None

        if dirichletBC == None:
            dirichletBC = self.dirichletBC
        else:
            self.dirichletBC = dirichletBC
        if neumanBC == None:
            neumanBC = self.neumanBC
        else:
            self.neumanBC = neumanBC


        self.K,self.b = self._construct(dirichletBC,neumanBC)                 #unscaled 5-point stencil
        self.K = self.K.tocsr()

    def solve(self, relaxation=0.8):
        """
        Solve A u = b.

        Parameters
        ----------
        dirElseNeu : bool
            True  → return Dirichlet traces u on [bottom, left, top, right]
            False → return Neumann traces (∂u/∂n) as (u_inner - u_bd)/dx

        Returns
        -------
        u : (N_x*N_y,) ndarray
        sideValues : list of 1D ndarrays (order: [bottom, left, top, right])
        """
        u = sp.linalg.spsolve(self.K, self.b).ravel()*relaxation

        #sideValues = []
        neumanns=[]
        dirichlets=[]
        for s in range(4):
            u_bd = u[self.ranges[0][s]]
            u_in = u[self.ranges[1][s]]
            neumanns.append((u_in - u_bd) / self.dx)
            dirichlets.append(u[self.ranges[0][s]])
        # if dirElseNeu:
        #     for s in range(4):           #Dirichlet traces (values on the boundary)
        #         sideValues.append(u[self.ranges[0][s]])
        # else:
        #     for s in range(4):    #Neumann traces: first-order difference consistent with outward normals
        #         u_bd = u[self.ranges[0][s]]
        #         u_in = u[self.ranges[1][s]]
        #         sideValues.append((u_in - u_bd) / self.dx)

        return u, {"neumann":neumanns, "dirichlet":dirichlets}
