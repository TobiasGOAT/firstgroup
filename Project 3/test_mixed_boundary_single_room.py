import numpy as np
from subdomain import Subdomain, Box, BCValues

def main():
    h = 0.05
    sd = Subdomain("Omega", Box(0.0, 0.0, 1.0, 1.0), h, BCValues())
    sd.assemble_matrix()

    Ny, Nx = sd.Ny, sd.Nx
    # Left side: bottom half Dirichlet=20, top half Neumann=g=0.3
    left_kind = np.full(Ny, "D", dtype="<U1")
    left_vals = np.full(Ny, 20.0, dtype=float)
    mid = Ny // 2
    left_kind[mid:] = "N"
    left_vals[mid:] = 0.3

    sd.apply_boundary_conditions(
        outer_dirichlet={"left": 0.0, "right": 15.0, "bottom": 15.0, "top": 15.0},
        interface_specs={
            "LeftMixed": {"side": "left", "kind": left_kind, "values": left_vals}
        }
    )
    sd.solve_local()
    U = sd.field_full()

    # Basic sanity: temps finite and within a reasonable envelope
    assert np.isfinite(U).all()
    mn, mx = float(np.min(U)), float(np.max(U))
    print(f"single-room mixed BCs: range [{mn:.3f}, {mx:.3f}]")
    assert -100.0 < mn < 100.0 and -100.0 < mx < 100.0
    print("OK")

if __name__ == "__main__":
    main()
