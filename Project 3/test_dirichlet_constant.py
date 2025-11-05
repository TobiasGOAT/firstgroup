import numpy as np
from subdomain import Subdomain, Box, BCValues

def main():
    box = Box(0.0, 0.0, 1.0, 1.0)
    h = 0.1
    sd = Subdomain("Omega", box, h, BCValues())
    sd.assemble_matrix()
    T = 20.0
    sd.apply_boundary_conditions(
        outer_dirichlet={"left": T, "right": T, "bottom": T, "top": T},
        interface_specs={}
    )
    sd.solve_local()
    U = sd.field_full()
    err = float(np.max(np.abs(U - T)))
    print("max error:", err)
    assert err < 1e-10, "Dirichlet-constant box should be exactly constant."

if __name__ == "__main__":
    main()
