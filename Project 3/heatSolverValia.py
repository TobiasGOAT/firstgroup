import numpy as np
import scipy.sparse as sp

"""
   A finite-difference solver for the steady-state 2D Laplace equation.

   Parameters
   ----------
   dx : float
       Grid spacing in both x and y directions. Defines the distance between
       neighboring grid points and is used in all derivative approximations.

   sides : list or tuple of floats
       [Lx, Ly] — the physical size of the rectangular domain in the x and y directions.
       For example, if Lx = 1.0 and Ly = 0.5, the grid will cover 0 ≤ x ≤ 1.0
       and 0 ≤ y ≤ 0.5.

   dirichletBC : list of lists (or None)
       [bottom, left, top, right] — values of the temperature `u` on the sides
       where **Dirichlet boundary conditions** are applied (fixed temperature).
       Each side is given as a list (or NumPy array) of values corresponding to
       each grid point along that side. The list length must match the number
       of grid points along that side (Nx or Ny). If a side has no Dirichlet
       boundary, write `None`.

   neumanBC : list of lists (or None)
       [bottom, left, top, right] — values of the Neumann boundary conditions
       (normal heat flux, ∂u/∂n) along each side. These describe the heat flow
       across the boundary instead of the temperature value. Like `dirichletBC`,
       each side is a list of flux values of the correct length, or `None` if
       no Neumann boundary is applied.

   Notes
   -----
   - The coordinate orientation is:
       bottom → y = 0
       top    → y = Ly
       left   → x = 0
       right  → x = Lx
   - The sides must not overlap: use either Dirichlet or Neumann per side.
   - Internally, the solver constructs a sparse matrix for the Laplacian using
     a 5-point stencil and solves `A u = b` for the steady-state temperature field.
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

