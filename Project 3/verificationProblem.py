import numpy as np
import matplotlib.pyplot as plt
from heatSolver import HeatSolver   # <-- replace with your file name

# ---------------------------------------------------------------------
# 1. Manufactured (exact) solution: u(x,y) = u_left + (u_right - u_left)/Lx * x
#     → Laplace(u)=0, Dirichlet at x=0,2, and zero-Neumann at y=0,1
# ---------------------------------------------------------------------
Lx, Ly = 2.0, 1.0
dx = 0.1
u_left, u_right = 0.0, 1.0

Nx = int(Lx/dx) + 1
Ny = int(Ly/dx) + 1
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing="xy")

# Exact analytic solution
u_exact = u_left + (u_right - u_left) / Lx * X

# ---------------------------------------------------------------------
# 2. Define BC arrays (matching solver convention)
#     Order: [bottom, left, top, right]
# ---------------------------------------------------------------------
dirichletBC = [
    None,                         # bottom (Neumann)
    np.full(Ny, u_left),          # left
    None,                         # top (Neumann)
    np.full(Ny, u_right)          # right
]

neumanBC = [
    np.zeros(Nx),  # bottom (∂u/∂n = 0)
    None,          # left
    np.zeros(Nx),  # top (∂u/∂n = 0)
    None           # right
]

# ---------------------------------------------------------------------
# 3. Build and solve
# ---------------------------------------------------------------------
solver = HeatSolver(dx=dx, sides=[Lx, Ly],
                    dirichletBC=dirichletBC,
                    neumanBC=neumanBC)

u_num_flat, sides = solver.solve(relaxation=1.0)
u_num = u_num_flat.reshape(Ny, Nx)

# ---------------------------------------------------------------------
# 4. Compare numerical vs. exact
# ---------------------------------------------------------------------
error = u_num - u_exact
max_err = np.abs(error).max()
mean_err = np.abs(error).mean()

print("===== VERIFICATION RESULTS =====")
print(f"Grid: {Nx} x {Ny} (dx={dx})")
print(f"Max error  : {max_err:.3e}")
print(f"Mean error : {mean_err:.3e}")

# ---------------------------------------------------------------------
# 5. Optional visualization
# ---------------------------------------------------------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.title("Numerical solution")
plt.pcolormesh(X, Y, u_num, shading='auto')
plt.colorbar(label='u')

plt.subplot(1,2,2)
plt.title("Error = u_num - u_exact")
plt.pcolormesh(X, Y, error, shading='auto', cmap='coolwarm')
plt.colorbar(label='Error')

plt.tight_layout()
plt.show()
