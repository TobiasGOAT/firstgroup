import numpy as np
import scipy.sparse as sp

"""
------------------------------------------------------------
READ ME FIRST
------------------------------------------------------------
A finite-difference solver for the 2D Laplace equation on a rectangle.

It builds a sparse matrix with the 5-point stencil,
applies the boundary conditions, and solves A u = b to get
the temperature at every grid point.
------------------------------------------------------------
y=Ly  ┌──────── top ────────┐
      │                     │
      │                     │
left  │                     │  right
x=0   │                     │  x=Lx
      │                     │
      └────── bottom ───────┘  y=0

------------------------------------------------------------
Parameters
------------------------------------------------------------
1. dx: float
2. sides: [Lx, Ly]
    Physical lengths. Grid sizes become:
      Nx = int(Lx/dx) + 1
      Ny = int(Ly/dx) + 1
3. dirichletBC: [bottom, left, top, right]
    Each entry is either:
      - a list/array of values of length Nx (for bottom/top) or Ny (for left/right)
      - or None if that side is not Dirichlet.
4. neumanBC: [bottom, left, top, right]
    Each entry is either:
      - a list/array of flux values with the same lengths as above
      - or None if that side is not Neumann.
"""


class heatSolver:

    def __init__(self,dx,sides,dirichletBC,neumanBC):
        #Sides are [L_x,L_y]
        #dirichletBC are [bottom,left,top,right]
        #neumanBC are [bottom,left,top,right], write None if no neuman BC
        #The values should be a list and must be N_x or N_y long respectively in order left to right or down to up

        self.dx = dx
        self.N_x = int(sides[0]/dx) + 1
        self.N_y = int(sides[1]/dx) + 1

        #Construct K matrix (A in slides)
        self.K_size = self.N_x*self.N_y
        self._constructKMat()   #build Laplacian matrix
        self.K = self.K / self.dx ** 2   #scale by 1/h^2
        self._createRanges()   #find and store the indices of all boundary grid points and their neighbouring inner points


        #Construct b (Au=b)
        self.b = sp.lil_matrix((self.K_size,1))

        self._applyBC(dirichletBC[0],neumanBC[0],self.ranges[0][0])   #Bottom
        self._applyBC(dirichletBC[1],neumanBC[1],self.ranges[0][1])   #left
        self._applyBC(dirichletBC[2],neumanBC[2],self.ranges[0][2])   #top
        self._applyBC(dirichletBC[3],neumanBC[3],self.ranges[0][3])   #right
        self.K = self.K.tocsr()   #convert to CSR

    def _createRanges(self):
        #In case the grids are too small
        if self.N_x < 2 or self.N_y < 2:
            raise ValueError("You messed up. You need N_x >= 2 and N_y >= 2 to define inner nodes.")

        #We map 2D grid indices (i,j) to 1D "flat" index k using row-major order: k = j*N_x + i
        Nx, Ny = self.N_x, self.N_y
        N = self.K_size  #=Nx * Ny

        #Boundary Indices
        bottom = np.arange(0, Nx)
        left = np.arange(0, N, Nx)
        top = np.arange(N - Nx, N)
        right = np.arange(Nx - 1, N, Nx)

        #Inner indices
        bottom_in = bottom + Nx
        left_in = left + 1
        top_in = top - Nx
        right_in = right - 1

        #store as both ordered lists and a named dict (pick one style if you prefer)
        self.ranges = [
            [bottom, left, top, right],
            [bottom_in, left_in, top_in, right_in]
        ]
        self.range_map = {
            "bottom": (bottom, bottom_in),
            "left": (left, left_in),
            "top": (top, top_in),
            "right": (right, right_in),
        }

    def _constructKMat(self):
        main = -4 * np.ones(self.K_size)
        LeftNRight = np.ones(self.K_size-1)
        LeftNRight[np.arange(1, self.N_y)*self.N_x - 1] = 0  # remove wrap-around
        UpNDown = np.ones(self.K_size - self.N_x)
        self.K = sp.diags([main, LeftNRight, LeftNRight, UpNDown, UpNDown], [0, 1, -1, self.N_x, -self.N_x], format='lil')

    def _applyBC(self,DBCList,NBCList,itterations):
        #Neuman
        list_index = 0
        if NBCList is not None:
            for i in itterations:
                self.K[i,i] += 1    
                self.b[i,0] += -NBCList[list_index]/self.dx
                list_index += 1
        # Dirichlet
        list_index = 0
        if DBCList is not None:
            for i in itterations:
                self.K[i, :] = 0
                self.K[i, i] = 1
                self.b[i, 0] = DBCList[list_index]  # <- change here
                list_index += 1

    def solve(self,dirElseNeu):
        #Gives value on boundaries in order bottom,left,top,right
        #dirElseNeu is if output should be dirichlet or neuman as boolean
        sol =sp.linalg.spsolve(self.K,self.b)
        sideValues = []

        if dirElseNeu:
            for i in range(0,4):
                sideValues.append(sol[self.ranges[0][i]])
        else:
            for i in range(0,4):
                sideValues.append((sol[self.ranges[1][i]] - sol[self.ranges[0][i]])/self.dx)

        return sol,sideValues

