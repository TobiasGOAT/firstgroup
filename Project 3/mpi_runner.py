# ============================================
# Run with:
#   mpiexec -n 3 python mpi_runner.py
# ============================================

from mpi4py import MPI
from geometry import Apartment
import numpy as np
import matplotlib.pyplot as plt
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 3:
    if rank == 0:
        print(" Size=3 required. (mpiexec -n 3 python mpi_runner.py)")
    exit()

# --- Geometry and setup ---
apt = Apartment(layout="default", dx=1/20)
rooms = apt.rooms
my_room = rooms[rank]

relax = 0.8
max_iter = 20
tol = 1e-3

for it in range(max_iter):
    # --- Boundary exchange ---
    if rank == 0:
        # Ω1 → send right Dirichlet to Ω2
        dirichlet_right = my_room.get_boundary_value("right", 0.0, 1.0)
        comm.send(dirichlet_right, dest=1, tag=10)
        # Ω1 ← receive Neumann correction from Ω2
        neumann_right = comm.recv(source=1, tag=20)
        my_room.N[3] = -neumann_right

    elif rank == 2:
        # Ω3 → send left Dirichlet to Ω2
        dirichlet_left = my_room.get_boundary_value("left", 0.0, 1.0)
        comm.send(dirichlet_left, dest=1, tag=30)
        # Ω3 ← receive Neumann correction from Ω2
        neumann_left = comm.recv(source=1, tag=40)
        my_room.N[1] = -neumann_left

    elif rank == 1:
        # Ω2 ← receive Dirichlet from Ω1 & Ω3
        left_D = comm.recv(source=0, tag=10)
        right_D = comm.recv(source=2, tag=30)
        my_room.D[1] = left_D
        my_room.D[3] = right_D

    # --- Local solve ---
    my_room.solver.updateBC(my_room.D, my_room.N)
    new_u, _ = my_room.solver.solve()
    my_room.u = relax * new_u + (1 - relax) * my_room.u

    # --- Send traces for Neumann update ---
    if rank == 1:
        left_trace = my_room.get_boundary_value("left", 0.0, 1.0)
        right_trace = my_room.get_boundary_value("right", 1.0, 2.0)
        comm.send(left_trace, dest=0, tag=20)
        comm.send(right_trace, dest=2, tag=40)

    # --- Check convergence ---
    local_res = np.max(np.abs(my_room.u - new_u))
    res = comm.allreduce(local_res, op=MPI.MAX)
    if rank == 0 and it % 2 == 0:
        print(f"Iteration {it:2d} | Global residual = {res:.3e}")
    if res < tol:
        if rank == 0:
            print(f"[OK] Converged after {it} iterations.")
        break




# --- Visualization ---

# --- Gather results to Rank 0 ---
u_data = comm.gather(my_room.u, root=0)
Nx_data = comm.gather(my_room.Nx, root=0)
Ny_data = comm.gather(my_room.Ny, root=0)

if rank == 0:
    print("\n Gathering completed. Plotting global temperature field...")

    # unpack sizes
    Ny0, Ny1, Ny2 = Ny_data[0], Ny_data[1], Ny_data[2]
    Nx0, Nx1, Nx2 = Nx_data[0], Nx_data[1], Nx_data[2]

    # reshape gathered 1D arrays into 2D fields
    u1 = u_data[0].reshape(Ny0, Nx0)  # Omega_1 (left)
    u2 = u_data[1].reshape(Ny1, Nx1)  # Omega_2 (middle)
    u3 = u_data[2].reshape(Ny2, Nx2)  # Omega_3 (right-bottom)

    # global canvas
    Y = max(Ny_data)
    X = sum(Nx_data) - 2  # two shared interface columns
    full = np.zeros((Y, X))

    # place Omega_1 (top-left)
    full[:Ny0, :Nx0] = u1

    # place Omega_2 (middle), overlapping 1 col with Omega_1
    full[:, Nx0-1 : Nx0+Nx1-1] = u2

    # place Omega_3 (bottom-right), overlapping 1 col with Omega_2
    x3_start = Nx0 - 2 + Nx1                 # == (Nx0-1) + (Nx1-1)
    full[Y-Ny2:, x3_start : x3_start+Nx2] = u3

    # indices of the interface columns
    border_L = Nx0 - 1               # Ω1 | Ω2
    border_R = Nx0 + Nx1 - 2         # Ω2 | Ω3

    # --- averaging borders ---
    # left border: average between last column of Ω1 and first column of Ω2
    full[:Ny0, border_L] = 0.5 * (u1[:Ny0, -1] + u2[:Ny0, 0])

    # right border: average between last column of Ω2 and first column of Ω3
    full[Y - Ny2:, border_R] = 0.5 * (u2[Y - Ny2:, -1] + u3[:, 0])

    # --- Save and plot  ---
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "temperature_field.png")

    # unified axes and colormap/range
    Lx, Ly = 3.0, 2.0
    VMIN, VMAX = 0.0, 40.0
    CMAP = "viridis"

    plt.figure(figsize=(8, 4.5))
    plt.imshow(
        full,
        origin="lower",
        extent=[0, Lx, 0, Ly], 
        aspect="equal",
        cmap=CMAP, vmin=VMIN, vmax=VMAX,
        interpolation="bilinear"  # nearest
    )
    cbar = plt.colorbar()
    cbar.set_label("Temperature (°C)")
    plt.xlabel("x"); plt.ylabel("y")
    plt.title("MPI Heat Distribution")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f" Figure saved to {out_path}")
