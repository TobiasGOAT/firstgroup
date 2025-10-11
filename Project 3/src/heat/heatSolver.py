import numpy as np
import scipy.sparse as sp


class heatSolver:
    """
    Steady-state heat (Laplace) equation on a rectangular grid using a
    five-point finite-difference stencil on a uniform Cartesian mesh.

    Each side (bottom, left, top, right) can have Dirichlet (temperature)
    or Neumann (flux) BCs. If both are given on a side, Dirichlet wins.
    Neumann flux is defined positive when heat is leaving the domain
    (outward normal derivative is positive).

    Parameters
    ----------
    dx : float
        Grid spacing in both directions (assumed identical in x and y).
    sides : (Lx, Ly)
        Physical dimensions. Number of nodes: Nx = int(Lx/dx)+1, Ny = int(Ly/dx)+1.
    dirichletBC : list of 4 lists/arrays or None
        [bottom, left, top, right] Dirichlet values per boundary node, or None.
        Ordering: bottom/top = left→right, left/right = bottom→top.
    neumannBC : list of 4 lists/arrays or None
        [bottom, left, top, right] Neumann fluxes per boundary node, or None.

    Notes
    -----
    The Laplacian matrix is first built unscaled, then scaled by 1/dx^2.
    Boundary rows are overwritten appropriately for Dirichlet/Neumann.
    """

    def __init__(self, dx, sides, dirichletBC, neumannBC):
        # Grid metrics
        self.dx = float(dx)
        self.N_x = int(sides[0] / self.dx) + 1
        self.N_y = int(sides[1] / self.dx) + 1
        self.K_size = self.N_x * self.N_y

        # Build unscaled Laplacian, then scale by 1/dx^2
        self._constructKMat()
        self.K = self.K / (self.dx ** 2)

        # Precompute boundary and first interior neighbour indices for each side
        self._createRanges()

        # RHS vector (lil for easy row overwrites)
        self.b = sp.lil_matrix((self.K_size, 1))

        # Apply BCs side-by-side: 0=bottom, 1=left, 2=top, 3=right
        for side in range(4):
            dbc = dirichletBC[side]
            nbc = neumannBC[side]
            boundary_nodes  = self.ranges[0][side]
            neighbour_nodes = self.ranges[1][side]
            self._applyBC(dbc, nbc, boundary_nodes, neighbour_nodes, side)

    def _createRanges(self):
        """
        Build index lists for boundary nodes and their first interior neighbours.

        Index flattening: k = j*N_x + i  with  i in [0..N_x-1], j in [0..N_y-1].
        """
        self.ranges = [[], []]

        # Bottom (y=0): boundary nodes (j=0), neighbours (j=1)
        bottom_boundary  = list(range(0, self.N_x))
        bottom_neighbour = [k + self.N_x for k in bottom_boundary]
        self.ranges[0].append(bottom_boundary)
        self.ranges[1].append(bottom_neighbour)

        # Left (x=0): boundary nodes (i=0), neighbours (i=1)
        left_boundary  = list(range(0, self.K_size, self.N_x))
        left_neighbour = [k + 1 for k in left_boundary]            # ← FIX: use left boundary, not bottom
        self.ranges[0].append(left_boundary)
        self.ranges[1].append(left_neighbour)

        # Top (y=Ly): boundary nodes (j=Ny-1), neighbours (j=Ny-2)
        top_boundary  = list(range(self.K_size - self.N_x, self.K_size))
        top_neighbour = [k - self.N_x for k in top_boundary]
        self.ranges[0].append(top_boundary)
        self.ranges[1].append(top_neighbour)

        # Right (x=Lx): boundary nodes (i=Nx-1), neighbours (i=Nx-2)
        right_boundary  = list(range(self.N_x - 1, self.K_size, self.N_x))
        right_neighbour = [k - 1 for k in right_boundary]
        self.ranges[0].append(right_boundary)
        self.ranges[1].append(right_neighbour)

    def _constructKMat(self):
        """Build the unscaled Laplacian (five-point stencil) in LIL format."""
        main_diag = -4.0 * np.ones(self.K_size)

        # Left/right neighbours (avoid row wrap-around)
        lr = np.ones(self.K_size - 1)
        row_end_indices = (np.arange(1, self.N_y) * self.N_x) - 1
        lr[row_end_indices] = 0.0

        # Up/down neighbours (always valid)
        ud = np.ones(self.K_size - self.N_x)

        self.K = sp.diags(
            [main_diag, lr, lr, ud, ud],
            [0, 1, -1, self.N_x, -self.N_x],
            format="lil"
        )

    def _applyBC(self, DBCList, NBCList, boundary_nodes, neighbour_nodes, side):
        """
        Overwrite rows for Dirichlet or Neumann on a given side.

        Parameters
        ----------
        DBCList : array-like or None
            Dirichlet values (len must equal len(boundary_nodes)) or None.
        NBCList : array-like or None
            Neumann fluxes (len must equal len(boundary_nodes)) or None.
            Positive means outward heat flux (leaving the domain).
        boundary_nodes : list[int]
            Flattened indices of boundary nodes.
        neighbour_nodes : list[int]
            Flattened indices of first interior neighbours.
        side : int
            0=bottom, 1=left, 2=top, 3=right.

        Notes
        -----
        If both Dirichlet and Neumann are provided, Dirichlet takes precedence.

        Finite-difference outward normal derivative conventions used:
          * bottom (y=0), left (x=0):   (u_bnd - u_nei) / dx = q
          * top (y=Ly), right (x=Lx):   (u_nei - u_bnd) / dx = q
        """
        # Length checks
        if DBCList is not None and len(DBCList) != len(boundary_nodes):
            raise ValueError(
                f"Dirichlet BC length {len(DBCList)} != number of boundary nodes {len(boundary_nodes)} on side {side}"
            )
        if NBCList is not None and len(NBCList) != len(boundary_nodes):
            raise ValueError(
                f"Neumann BC length {len(NBCList)} != number of boundary nodes {len(boundary_nodes)} on side {side}"
            )

        # Dirichlet has priority
        if DBCList is not None:
            for idx, node in enumerate(boundary_nodes):
                value = float(DBCList[idx])
                self.K[node, :] = 0.0
                self.K[node, node] = 1.0
                self.b[node, 0] = value
            return

        # Neumann (if any)
        if NBCList is not None:
            for idx, node in enumerate(boundary_nodes):
                q = float(NBCList[idx])
                nei = neighbour_nodes[idx]
                self.K[node, :] = 0.0
                if side in (0, 1):  # bottom or left: outward normal negative
                    # (u_bnd - u_nej)/dx = q
                    self.K[node, node] =  1.0 / self.dx
                    self.K[node, nei]  = -1.0 / self.dx
                    self.b[node, 0]    =  q
                else:               # top or right: outward normal positive
                    # (u_nej - u_bnd)/dx = q
                    self.K[node, node] = -1.0 / self.dx
                    self.K[node, nei]  =  1.0 / self.dx
                    self.b[node, 0]    =  q

    def solve(self, returnDirichlet=True):
        """
        Solve the linear system and return the solution + boundary values.

        Parameters
        ----------
        returnDirichlet : bool
            If True: return boundary temperatures; else boundary fluxes.

        Returns
        -------
        sol : (N_x*N_y,) ndarray
            Flattened temperature field.
        sideValues : list of 4 ndarrays
            Values on [bottom, left, top, right]. Either temperature (if
            returnDirichlet) or outward normal derivative (flux) otherwise.
        """
        # Solve (convert K to CSR for speed)
        sol = sp.linalg.spsolve(self.K.tocsr(), self.b)
        sol = np.asarray(sol).flatten()

        sideValues = []
        if returnDirichlet:
            # Just pick the boundary nodes
            for i in range(4):
                idx = np.fromiter(self.ranges[0][i], dtype=int)
                sideValues.append(sol[idx])
        else:
            # Compute outward normal derivatives using proper neighbour pairs
            for i in range(4):
                bnd = np.fromiter(self.ranges[0][i], dtype=int)
                nei = np.fromiter(self.ranges[1][i], dtype=int)
                if i in (0, 1):  # bottom or left: outward normal negative
                    flux = (sol[bnd] - sol[nei]) / self.dx
                else:            # top or right: outward normal positive
                    flux = (sol[nei] - sol[bnd]) / self.dx
                sideValues.append(flux)

        return sol, sideValues
