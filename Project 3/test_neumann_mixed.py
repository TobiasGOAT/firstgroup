# Bottom = Neumann(g), others = Dirichlet; then check recovered normal derivative.
import numpy as np
from subdomain import Subdomain, Box, BCValues

def u_exact(x, y, a=0.8, b=0.6, c=1.0):
    return a * x + b * y + c  # Laplacian = 0

def main():
    a, b, c = 0.8, 0.6, 1.0
    box = Box(0.0, 0.0, 1.0, 1.0)
    h = 0.05
    sd = Subdomain("Omega", box, h, BCValues())
    sd.assemble_matrix()

    left   = u_exact(sd.x[0],  sd.y,    a, b, c)
    right  = u_exact(sd.x[-1], sd.y,    a, b, c)
    top    = u_exact(sd.x,     sd.y[-1],a, b, c)

    # On bottom, outward normal points toward -y, so g = ∂u/∂n = -uy = -b
    g_bottom = np.full(sd.Nx, -b, dtype=float)

    sd.apply_boundary_conditions(
        outer_dirichlet={"left": 0, "right": 0, "bottom": 0, "top": 0},
        interface_specs={
            "L": {"type": "dirichlet", "side": "left",   "values": left},
            "R": {"type": "dirichlet", "side": "right",  "values": right},
            "T": {"type": "dirichlet", "side": "top",    "values": top},
            "B": {"type": "neumann",   "side": "bottom", "values": g_bottom},
        }
    )
    sd.solve_local()
    g_hat = sd.extract_normal_derivative("bottom")
    err = float(np.max(np.abs(g_hat - g_bottom)))
    print("max |g_hat - g|:", err)
    assert err < 1e-2, "Bottom Neumann derivative should be matched within tolerance."

if __name__ == "__main__":
    main()
