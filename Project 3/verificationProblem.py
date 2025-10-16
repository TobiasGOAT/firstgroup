import numpy as np
import matplotlib.pyplot as plt
from heatSolver import HeatSolver   # <-- replace with your file name


# ---------------------------------------------------------------------
# Domain and parameters
# ---------------------------------------------------------------------
Lx, Ly = 2.0, 1.0
dx = 0.05

Nx = int(Lx/dx) + 1
Ny = int(Ly/dx) + 1
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing="xy")

# ---------------------------------------------------------------------
# Manufactured exact solution (harmonic)
# ---------------------------------------------------------------------
def u_exact_fn(x, y):
    return np.sin(np.pi * x / Lx) * np.sinh(np.pi * y / Lx)

u_exact = u_exact_fn(X, Y)

# ---------------------------------------------------------------------
# Compute boundary data
# ---------------------------------------------------------------------
# Dirichlet (vertical)
u_left = u_exact_fn(0, y)
u_right = u_exact_fn(Lx, y)

# Neumann (horizontal) -> outward normal derivative
def u_y(x, y):
    return (np.pi / Lx) * np.sin(np.pi * x / Lx) * np.cosh(np.pi * y / Lx)

q_bottom = u_y(x, 0.0)           # ∂u/∂n at y=0
q_top = u_y(x, Ly)               # ∂u/∂n at y=Ly

# ---------------------------------------------------------------------
# Define BC arrays for solver
#   Order: [bottom, left, top, right]
# ---------------------------------------------------------------------
dirichletBC = [
    None,              # bottom → Neumann
    u_left,            # left Dirichlet
    None,              # top → Neumann
    u_right            # right Dirichlet
]

neumanBC = [
    q_bottom,          # bottom
    None,              # left
    q_top,             # top
    None               # right
]

# ---------------------------------------------------------------------
# Build solver and solve
# ---------------------------------------------------------------------
solver = HeatSolver(dx=dx, sides=[Lx, Ly],
                    dirichletBC=dirichletBC,
                    neumanBC=neumanBC)

u_num_flat, sides = solver.solve(relaxation=1.0)
u_num = u_num_flat.reshape(Ny, Nx)

# ---------------------------------------------------------------------
# Compare numerical vs. exact
# ---------------------------------------------------------------------
error = u_num - u_exact
max_err = np.abs(error).max()
mean_err = np.abs(error).mean()

print("===== VERIFICATION RESULTS (nonhomogeneous Neumann) =====")
print(f"Grid: {Nx} x {Ny}  (dx={dx})")
print(f"Max error  : {max_err:.3e}")
print(f"Mean error : {mean_err:.3e}")

# ---------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------
plt.figure(figsize=(11,4))
plt.subplot(1,2,1)
plt.title("Numerical solution u_num")
plt.pcolormesh(X, Y, u_num, shading='auto')
plt.colorbar(label='u')

plt.subplot(1,2,2)
plt.title("Error = u_num - u_exact")
plt.pcolormesh(X, Y, error, shading='auto', cmap='coolwarm')
plt.colorbar(label='Error')
plt.tight_layout()
plt.show()
