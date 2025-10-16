import numpy as np
from scipy.linalg import solve


class Room:
    """
    Class representing a 2D room for heat distribution simulation.
    """

    def __init__(self, delta, size_x, size_y):
        """Initialize the Room class.

        Args:
            delta_x (float): The spatial step size in the x direction.
            size_x (float): The physical size of the room in the x direction.
            size_y (float): The physical size of the room in the y direction.
        """
        self.delta = delta
        self.Nx = int(size_x * (1 / delta) + 1)  # Number of grid points in x direction
        self.Ny = int(size_y * (1 / delta) + 1)  # Number of grid points in y direction
        self.Nx_int = self.Nx - 2  # Internal grid points in x direction
        self.Ny_int = self.Ny - 2  # Internal grid points in y direction
        self.N = self.Nx_int * self.Ny_int  # Total internal grid points
        self.dx2 = delta**2

    def index(self, i, j):
        """Get the linear index for a 2D grid point.

        Args:
            i (int): The x-coordinate (column index) of the grid point.
            j (int): The y-coordinate (row index) of the grid point.

        Returns:
            int: The linear index corresponding to the 2D grid point (i, j).
        """
        return i * self.N + j

    def A_dirichlet(self):
        """Construct the coefficient matrix A for the Dirichlet conditions.

        Returns:
            np.ndarray: The coefficient matrix A.
        """
        A = np.zeros((self.N, self.N))

        k = 0  # index for internal points
        for i in range(1, self.Nx - 1):
            for j in range(1, self.Ny - 1):
                A[k, k] = -4 / self.dx2

                if i > 1:  # check if not on left boundary
                    A[k, k - 2] = 1 / self.dx2  # left neighbor
                if i < self.Nx - 2:
                    A[k, k + 2] = 1 / self.dx2  # right neighbor
                if j > 1:
                    A[k, k - 1] = 1 / self.dx2  # bottom neighbor
                if j < self.Ny - 2:
                    A[k, k + 1] = 1 / self.dx2  # top neighbor
                k += 1
        return A

    def A_neumann(self, neumann_side="right"):
        """Construct the coefficient matrix A for the Neumann conditions.

        Returns:
            np.ndarray: The coefficient matrix A.
        """
        if neumann_side not in ("left", "right"):
            raise ValueError("neumann_side must be 'left' or 'right'")
        A = np.zeros((self.N, self.N))
        k = 0
        for i in range(1, self.Nx - 1):
            for j in range(1, self.Ny - 1):
                A[k, k] = -4 / self.dx2

                if i > 1:
                    A[k, k - self.Ny_int] = 1 / self.dx2  # left neighbor
                elif (
                    neumann_side == "left" and i == 1
                ):  # Neumann: replace missing neighbor
                    A[k, k] += 1 / self.dx2  # adjust diagonal coefficoent from -4 to -3
                if i < self.Nx - 2:
                    A[k, k + self.Ny_int] = 1 / self.dx2  # right neighbor
                elif neumann_side == "right" and i == self.Nx - 2:
                    A[k, k] += 1 / self.dx2
                if j > 1:
                    A[k, k - 1] = 1 / self.dx2  # bottom neighbor
                if j < self.Ny - 2:
                    A[k, k + 1] = 1 / self.dx2  # top neighbor

                k += 1
        return A

    def b_dirichlet(self, t_left, t_right, t_bottom, t_top):
        """Construct the right-hand side vector b for the Dirichlet conditions.

        Args:
            t_left (np.ndarray): The temperature at the left boundary.
            t_right (np.ndarray): The temperature at the right boundary.
            t_bottom (np.ndarray): The temperature at the bottom boundary.
            t_top (np.ndarray): The temperature at the top boundary.

        Returns:
            np.ndarray: The right-hand side vector b.
        """

        b = np.zeros(self.N)
        # Boundary conditions
        k = 0  # index for internal points
        for i in range(1, self.Nx - 1):
            for j in range(1, self.Ny - 1):
                if i == 1:  # left (border)
                    b[k] -= t_left[j] / self.dx2
                if i == self.Nx - 2:  # right (border)
                    b[k] -= t_right[j] / self.dx2
                if j == 1:  # bottom (border)
                    b[k] -= t_bottom[i] / self.dx2
                if j == self.Ny - 2:  # top (border)
                    b[k] -= t_top[i] / self.dx2

                k += 1
        return b

    def b_dirichlet_two_by_one(
        self,
        t_topright,
        t_bottomleft,
        t_bottom=5,
        t_top=40,
        t_topleft=15,
        t_bottomright=15,
    ):
        """Construct the right-hand side vector b for the Dirichlet conditions.

        Args:
            t_bottom (float): The temperature at the bottom boundary.
            t_top (float): The temperature at the top boundary.
            t_topleft (float): The temperature at the top left boundary.
            t_topright (np.ndarray): The temperature at the topright boundary.
            t_bottomleft (np.ndarray): The temperature at the bottomleft boundary.
            t_bottomright (float): The temperature at the bottomright boundary.

        Returns:
            np.ndarray: The right-hand side vector b.
        """
        b = np.zeros(self.N)
        # Boundary conditions
        k = 0  # index for internal points
        midpoint = (self.Ny - 2) // 2 + 1
        for i in range(1, self.Nx - 1):
            for j in range(1, self.Ny - 1):
                if i == 1:  # left (border)
                    if j <= midpoint:
                        b[k] -= t_bottomleft[j - 1] / self.dx2
                    else:
                        b[k] -= t_topleft / self.dx2
                if i == self.Nx - 2:  # right (border)
                    if j < midpoint:
                        b[k] -= t_bottomright / self.dx2
                    else:
                        b[k] -= t_topright[j - midpoint - 1] / self.dx2
                if j == 1:  # bottom (border)
                    b[k] -= t_bottom / self.dx2
                if j == self.Ny - 2:  # top (border)
                    b[k] -= t_top / self.dx2

                k += 1
        return b

    def b_neumann(
        self,
        t_left,
        t_right,
        q_left,
        q_right,
        t_bottom=15,
        t_top=15,
    ):  # flux = q
        """Construct the right-hand side vector b for the Neumann conditions.

        Args:
            t_left (np.ndarray): The temperature at the left boundary.
            t_right (np.ndarray): The temperature at the right boundary.
            t_bottom (float): The temperature at the bottom boundary.
            t_top (float): The temperature at the top boundary.
            q_left (np.ndarray): The temperature at the left boundary.
            q_right (np.ndarray): The temperature at the right boundary.

        Returns:
            np.ndarray: The right-hand side vector b.
        """
        b = np.zeros(self.N)
        k = 0
        for i in range(1, self.Nx - 1):
            for j in range(1, self.Ny - 1):
                if i == 1:
                    if t_left is not None:  # left (Dirichlet border)
                        b[k] -= t_left[j] / self.dx2
                    elif q_left is not None:  # left (Neumann border)
                        b[k] += q_left[j] / self.delta_x
                if i == self.Nx - 2:
                    if t_right is not None:  # right (Dirichlet border)
                        b[k] -= t_right[j] / self.dx2
                    elif q_right is not None:  # right (Neumann border) -q/h
                        b[k] -= q_right[j] / self.delta_x
                if j == 1 and t_bottom is not None:  # bottom (border)
                    b[k] -= t_bottom / self.dx2
                if j == self.Ny - 2 and t_top is not None:  # top (border)
                    b[k] -= t_top / self.dx2
                k += 1
        return b

    def solve(self, A, b):
        """Solve the linear system Ax = b.

        Args:
            A (np.ndarray): The coefficient matrix.
            b (np.ndarray): The right-hand side vector.

        Returns:
            np.ndarray: The solution vector.
        """
        T = solve(A, b)
        return T  #
