import numpy as np
from subdomain import Subdomain, Box, BCValues

def u_exact(x, y, a=1.3, b=-0.7, c=2.0):
    return a * x + b * y + c

def main():
    a, b, c = 1.3, -0.7, 2.0
    box = Box(0.0, 0.0, 1.0, 0.8)
    h = 0.05
    sd = Subdomain("Omega", box, h, BCValues())
    sd.assemble_matrix()

    # Build Dirichlet arrays from the exact function on each side
    left   = u_exact(sd.x[0],  sd.y,    a, b, c)  # length Ny
    right  = u_exact(sd.x[-1], sd.y,    a, b, c)
    bottom = u_exact(sd.x,     sd.y[0], a, b, c)  # length Nx
    top    = u_exact(sd.x,     sd.y[-1],a, b, c)

    sd.apply_boundary_conditions(
        outer_dirichlet={"left": 0, "right": 0, "bottom": 0, "top": 0},  # overridden
        interface_specs={
            "L": {"type": "dirichlet", "side": "left",   "values": left},
            "R": {"type": "dirichlet", "side": "right",  "values": right},
            "B": {"type": "dirichlet", "side": "bottom", "values": bottom},
            "T": {"type": "dirichlet", "side": "top",    "values": top},
        }
    )
    sd.solve_local()
    U = sd.field_full()

    X, Y = np.meshgrid(sd.x, sd.y, indexing="xy")
    Uex = u_exact(X, Y, a, b, c)

    err = float(np.max(np.abs(U - Uex)))
    print("max error:", err)
    assert err < 1e-10, "Linear manufactured solution should be reproduced."

if __name__ == "__main__":
    main()
