import numpy as np
import matplotlib.pyplot as plt
from heatSolver import HeatSolver

def run_verification_with_plot(dx=0.05, Lx=2.0, Ly=1.0):
    Nx = int(Lx / dx) + 1
    Ny = int(Ly / dx) + 1
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    X, Y = np.meshgrid(x, y, indexing='xy')

    # --- Exact linear solution ---
    a = 0.7
    b = 1.3
    c = -0.4
    def u_exact_fn(x, y): return a + b*x + c*y

    u_exact = u_exact_fn(X, Y)

    # Dirichlet on left/right
    dir_left  = u_exact_fn(0.0, y)
    dir_right = u_exact_fn(Lx, y)

    # Neumann on bottom/top (outward normal)
    q_bottom = np.full(Nx, -c)
    q_top    = np.full(Nx,  c)

    dirichletBC = [None, dir_left, None, dir_right]
    neumanBC = [q_bottom, None, q_top, None]

    solver = HeatSolver(dx=dx, sides=[Lx, Ly],
                        dirichletBC=dirichletBC,
                        neumanBC=neumanBC)

    u_num_flat, sides_data = solver.solve()
    u_num = u_num_flat.reshape(Ny, Nx)

    # --- Compute error ---
    error = u_num - u_exact
    max_err = np.abs(error).max()
    mean_err = np.abs(error).mean()
    print("LINEAR VERIFICATION TEST WITH PLOTS")
    print(f"Max error  : {max_err:.3e}")
    print(f"Mean error : {mean_err:.3e}")

    # --- Plot numerical solution ---
    fig, axs = plt.subplots(1, 3, figsize=(18,5))
    cs1 = axs[0].contourf(X, Y, u_num, 20, cmap='viridis')
    axs[0].set_title("Numerical solution")
    fig.colorbar(cs1, ax=axs[0])

    # Plot exact solution
    cs2 = axs[1].contourf(X, Y, u_exact, 20, cmap='viridis')
    axs[1].set_title("Exact solution")
    fig.colorbar(cs2, ax=axs[1])

    # Plot error
    cs3 = axs[2].contourf(X, Y, error, 20, cmap='RdBu')
    axs[2].set_title("Error (u_num - u_exact)")
    fig.colorbar(cs3, ax=axs[2])

    for ax in axs:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()

    return solver, u_num, u_exact, error

if __name__ == "__main__":
    solver, u_num, u_exact, error = run_verification_with_plot(dx=0.05)
