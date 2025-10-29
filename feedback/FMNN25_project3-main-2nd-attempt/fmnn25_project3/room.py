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
        self.size_x = size_x
        self.size_y = size_y

    def compute_internal_points(self, neumann_left=False, neumann_right=False):
        """Compute the number of internal grid points in x and y directions,
        automatically adjusting for Neumann boundaries.

        Args:
            neumann_left (bool): True if Neumann condition on left boundary.
            neumann_right (bool): True if Neumann condition on right boundary.

        Updates:
            self.Nx_int, self.Ny_int, self.N, self.grid_info
        """
        # Compute total number of grid points (including boundaries)
        Nx_total = int(round(self.size_x / self.delta)) + 1
        Ny_total = int(round(self.size_y / self.delta)) + 1

        # Internal points (without outer boundaries)
        Nx_int = Nx_total - 2
        Ny_int = Ny_total - 2

        # Adjust for Neumann boundaries
        if neumann_left:
            Nx_int += 1
        if neumann_right:
            Nx_int += 1

        # Save to instance
        self.Nx_total = Nx_total
        self.Ny_total = Ny_total
        self.Nx_int_neumann = Nx_int
        self.Ny_int_neumann = Ny_int
        self.N_neumann = Nx_int * Ny_int

    def A_dirichlet(self):
        """Construct the coefficient matrix A for the Dirichlet conditions.

        Returns:
            np.ndarray: The coefficient matrix A.
        """
        A = np.zeros((self.N, self.N))

        k = 0  # index for internal points
        for j in range(1, self.Ny - 1):
            for i in range(1, self.Nx - 1):
                A[k, k] = -4 / self.dx2
                if i > 1:  # check if not on left boundary
                    A[k, k - 1] = 1 / self.dx2  # left neighbor
                if i < self.Nx - 2:
                    A[k, k + 1] = 1 / self.dx2  # right neighbor
                if j > 1:
                    A[k, k - self.Nx + 2] = 1 / self.dx2  # bottom neighbor
                if j < self.Ny - 2:
                    A[k, self.Nx - 2 + k] = 1 / self.dx2  # top neighbor
                k += 1
        return A

    def A_neumann(self, neumann_left=False, neumann_right=False):
        """Construct the coefficient matrix A for the Neumann conditions (row-major)."""

        self.compute_internal_points(neumann_left, neumann_right)
        A = np.zeros((self.N_neumann, self.N_neumann))
        k = 0
        for j in range(1, self.Ny_int_neumann + 1):
            for i in range(1, self.Nx_int_neumann + 1):
                A[k, k] = -4 / self.dx2

                if i > 1:  # left neighbor
                    A[k, k - 1] = 1 / self.dx2
                elif neumann_left and i == 1:
                    A[k, k] += 1 / self.dx2

                if i < self.Nx_int_neumann:  # right neighbor
                    A[k, k + 1] = 1 / self.dx2
                elif neumann_right and i == self.Nx_int_neumann:
                    A[k, k] += 1 / self.dx2

                if j > 1:
                    A[k, k - self.Nx_int_neumann] = 1 / self.dx2  # bottom neighbor
                if j < self.Ny_int_neumann:
                    A[k, k + self.Nx_int_neumann] = 1 / self.dx2  # top neighbor

                k += 1
        return A

    def b_dirichlet_two_by_one(
        self, gamma2, gamma1, T_bottom=5, T_top=40, T_left_top=15, T_right_bottom=15
    ):
        """Construct the right-hand side vector b for the Dirichlet conditions.

        Args:
            T_bottom (float): The temperature at the bottom boundary.
            T_top (float): The temperature at the top boundary.
            T_left_top (float): The temperature at the top left boundary.
            gamma2 (np.ndarray): The temperature at the topright boundary.
            gamma1 (np.ndarray): The temperature at the bottomleft boundary.
            T_right_bottom (float): The temperature at the bottomright boundary.
        Returns:
            np.ndarray: The right-hand side vector b.
        """
        b = np.zeros(self.N)
        # Boundary conditions
        k = 0  # index for internal points
        midpoint = (self.Ny - 2) // 2 + 1
        for j in range(1, self.Ny - 1):
            for i in range(1, self.Nx - 1):
                if i == 1:  # left (border)
                    if j < midpoint:
                        b[k] -= gamma1[j - 1] / self.dx2
                    else:
                        b[k] -= T_left_top / self.dx2
                if i == self.Nx - 2:  # right (border)
                    if j <= midpoint:
                        b[k] -= T_right_bottom / self.dx2
                    else:
                        b[k] -= gamma2[j - midpoint - 1] / self.dx2
                if j == 1:  # bottom (border)
                    b[k] -= T_bottom / self.dx2
                if j == self.Ny - 2:  # top (border)
                    b[k] -= T_top / self.dx2
                k += 1
        return b
    
    def b_dirichlet_two_by_one_4(
        self,
        gamma1,
        gamma2,
        gamma3,
        T_bottom=5,
        T_top=40,
        T_left_top=15,
        T_right_bottom=15
    ):
        """Construct the right-hand side vector b for the Dirichlet conditions.

        Args:
            T_bottom (float): The temperature at the bottom boundary.
            T_top (float): The temperature at the top boundary.
            T_left_top (float): The temperature at the top left boundary.
            gamma2 (np.ndarray): The temperature at the topright boundary.
            gamma1 (np.ndarray): The temperature at the bottomleft boundary.
            gamma3 (np.ndarray): The temperature at the bottomright boundary.
            T_right_bottom (float): The temperature at the bottomright boundary.
        Returns:
            np.ndarray: The right-hand side vector b.
        """
        b = np.zeros(self.N)
        # Boundary conditions
        k = 0  # index for internal points
        midpoint = (self.Ny - 2) // 2 + 1
        quarter = (self.Ny - 2) // 4
        for j in range(1, self.Ny - 1):
            for i in range(1, self.Nx - 1):
                if i == 1:  # left (border)
                    if j < midpoint:
                        b[k] -= gamma1[j - 1] / self.dx2
                    else:
                        b[k] -= T_left_top / self.dx2
                if i == self.Nx - 2:  # right (border)
                    if j <= quarter:
                        b[k] -= T_right_bottom / self.dx2
                    elif j < midpoint:
                        idx_gamma3 = j - quarter - 2
                        b[k] -= gamma3[idx_gamma3] / self.dx2
                    elif j == midpoint:
                        b[k] -= T_right_bottom/self.dx2
                    else:
                        idx_gamma2 = j - midpoint - 1
                        b[k] -= gamma2[idx_gamma2] / self.dx2
                if j == 1:  # bottom (border)
                    b[k] -= T_bottom / self.dx2
                if j == self.Ny - 2:  # top (border)
                    b[k] -= T_top / self.dx2

                k += 1
        return b

    def b_neumann(
        self, T_bottom, T_top, T_left=None, T_right=None, q_left=None, q_right=None
    ):

        self.compute_internal_points(
            neumann_left=(q_left is not None), neumann_right=(q_right is not None)
        )

        b = np.zeros(self.N_neumann)
        k = 0
        for j in range(1, self.Ny_int_neumann + 1):
            for i in range(1, self.Nx_int_neumann + 1):
                # Left boundary
                if i == 1:
                    if T_left is not None:  # Dirichlet
                        b[k] -= T_left / self.dx2
                    elif q_left is not None:  # Neumann
                        b[k] -= q_left[j - 1] / self.delta

                # Right boundary
                if i == self.Nx_int_neumann:
                    if T_right is not None:  # Dirichlet
                        b[k] -= T_right / self.dx2
                    elif q_right is not None:  # Neumann
                        b[k] -= q_right[j - 1] / self.delta

                # Bottom boundary
                if j == 1:
                    b[k] -= T_bottom / self.dx2

                # Top boundary
                if j == self.Ny - 2:
                    b[k] -= T_top / self.dx2

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


if __name__ == "__main__":
    dx = 1 / 3
    dx2 = dx**2
    T_normal = 15
    T_h = 40
    T_wf = 5

    q_right = np.array([10, 30])
    # Example usage
    room = Room(delta=1 / 3, size_x=1, size_y=2)
    b1 = room.b_dirichlet_two_by_one_4(
        gamma1= [0,0],
        gamma2 = [0,0],
        gamma3 = [0],
        T_bottom=5,
        T_top=40,
        T_left_top=15,
        T_right_bottom=15
    )
    
    print(b1)
    from .matrix import matrix

    print(
        matrix().b1(dx, T_bottom=T_normal, T_left=T_h, T_top=T_normal, q_right=q_right)
    )
