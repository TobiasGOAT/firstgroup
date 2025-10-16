from room import Room
import matplotlib.pyplot as plt
import numpy as np

class Apartment:
    def __init__(self, layout="default", dx=1/20):
        if layout=="default":
            omega1=Room(dx, (1, 1), ["left"], [])
            omega2=Room(dx, (1, 2), ["top"], ["bottom"])
            omega3=Room(dx, (1, 1), ["right"], [])
            coupling1_2={"neighbor": omega2, "side":"right", "start":0.0, "end":1.0}
            coupling2_1={"neighbor": omega1, "side": "left", "start": 0.0, "end": 1.0}
            coupling2_3={"neighbor": omega3, "side": "right", "start": 1.0, "end": 2.0}
            coupling3_2={"neighbor": omega2, "side": "left", "start": 0.0, "end": 1.0}
            omega1.add_coupling(coupling1_2)
            omega2.add_coupling(coupling2_1)
            omega2.add_coupling(coupling2_3)
            omega3.add_coupling(coupling3_2)
            self.rooms=[omega1, omega2, omega3]
        else:
            raise NotImplementedError("only default layout implemented")
        
    def plot(self):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3), constrained_layout=True)
        titles = ["Ω1", "Ω2", "Ω3"]
        vmin, vmax = 5.0, 40.0
        for ax, room, title in zip(axes, self.rooms, titles):
            field = room.u.reshape(room.Ny, room.Nx)
            x = np.linspace(0, room.Lx, room.Nx)
            y = np.linspace(0, room.Ly, room.Ny)
            im = ax.contourf(x, y, field, levels=50, vmin=vmin, vmax=vmax)
            ax.set_title(title); ax.set_xlabel(""); ax.set_ylabel("")
        cbar = fig.colorbar(im, ax=axes.ravel().tolist())
        cbar.set_label("Temperature")
        plt.show()
    
    # drop-in for geometry.Apartment

    def iterate(self, max_iter=200, omega=0.6, tol=1e-4, verbose=False, beta=0.1):
        """
        DN loop

        We split each interface as A|B. Here, B is the "Dirichlet side" (think Ω2),
        and A is the "Neumann side" (Ω1 or Ω3). We keep a guess for the temperature
        on B's interface segment, then iterate:
        1) Apply that Dirichlet guess on B and solve B.
        2) Read the heat flux from B on the interface and apply it (with opposite
            normal) as Neumann on A, then solve A.
        3) Read A's temperature trace on the same interface and relax our guess.
        4) Track the max change (residual). Stop when it’s small enough.

        This is intentionally straightforward (no grouping, no extra damping).
        """

        # --- tiny helpers we all know we’ll need ---
        side_to_id = {"bottom": 0, "left": 1, "top": 2, "right": 3}
        def opposite(side): return {"bottom":"top","top":"bottom","left":"right","right":"left"}[side]
        def side_len(room, side): return len(room.side_to_indices[side])
        def slice_idx(room, side, start_len, end_len):
            # map physical [start_len,end_len] on this side -> index slice
            idx = room.side_to_indices[side]
            i0 = int(round(start_len / room.dx))
            i1 = int(round(end_len   / room.dx))
            i0 = max(0, min(i0, len(idx)))
            i1 = max(0, min(i1, len(idx)))
            return idx[i0:i1], i0, i1

        # --- build interface list (each with both local ranges) ---
        # note to team: we assume reverse couplings exist on the neighbor.
        interfaces, seen = [], set()
        for A in self.rooms:
            for cA in A.neighbors:
                B = cA["neighbor"]; sideA = cA["side"]; sideB = opposite(sideA)
                rev = None
                for cB in B.neighbors:
                    if cB["neighbor"] is A and cB["side"] == sideB:
                        rev = cB; break
                if rev is None:
                    raise ValueError("Missing reverse coupling for an interface.")
                key = tuple(sorted([
                    (id(A), sideA, float(cA["start"]), float(cA["end"])),
                    (id(B), sideB, float(rev["start"]), float(rev["end"]))
                ]))
                if key in seen:  # avoid duplicates
                    continue
                seen.add(key)
                interfaces.append({
                    "A": A, "B": B,
                    "sideA": sideA, "sideB": sideB,
                    "startA": float(cA["start"]), "endA": float(cA["end"]),
                    "startB": float(rev["start"]), "endB": float(rev["end"]),
                })

        if verbose:
            print(f"[DN] Interfaces: {len(interfaces)}")

        # --- initial Dirichlet guesses on B’s side ---
        # if B already has a Dirichlet array, use that segment; otherwise fallback to normal wall temp.
        dirichlet_guess = {}
        for itf in interfaces:
            B, sideB = itf["B"], itf["sideB"]
            _, j0, j1 = slice_idx(B, sideB, itf["startB"], itf["endB"])
            sB = side_to_id[sideB]
            if B.D[sB] is not None:
                g = np.array(B.D[sB][j0:j1], dtype=float)
            else:
                g = np.full(j1 - j0, getattr(B, "normal_wall_temp", 15.0), dtype=float)
            dirichlet_guess[(id(itf["A"]), id(B), itf["sideA"], sideB)] = g

        last_res = float("inf")

        # --- main DN loop (one interface at a time) ---
        for k in range(max_iter):
            max_res = 0.0

            for itf in interfaces:
                A, B = itf["A"], itf["B"]
                sideA, sideB = itf["sideA"], itf["sideB"]

                # index ranges on both sides (we’ll use the min length to stay safe)
                _, i0, i1 = slice_idx(A, sideA, itf["startA"], itf["endA"])
                _, j0, j1 = slice_idx(B, sideB, itf["startB"], itf["endB"])
                lenA = i1 - i0; lenB = j1 - j0; m = min(lenA, lenB)
                if m <= 0:
                    continue

                key = (id(A), id(B), sideA, sideB)
                sA, sB = side_to_id[sideA], side_to_id[sideB]

                # 1) apply Dirichlet guess on B and solve B
                D_B = [None if x is None else x.copy() for x in B.D]
                N_B = [None if x is None else x.copy() for x in B.N]
                if D_B[sB] is None:
                    D_B[sB] = np.full(side_len(B, sideB), getattr(B, "normal_wall_temp", 15.0), dtype=float)
                g = dirichlet_guess[key]
                mm = min(m, len(g))              # safety: segment and guess might mismatch
                D_B[sB][j0:j0+mm] = g[:mm]
                N_B[sB] = None                   # Dirichlet wins on B’s interface side
                B.solver.updateBC(D_B, N_B)
                uB, _ = B.solver.solve(True)
                B.u = uB

                # 2) read flux from B and impose on A as Neumann (with opposite normal), then solve A
                _, neuB = B.solver.solve(False)
                flux_seg = np.asarray(neuB[sB][j0:j0+mm], dtype=float)

                D_A = [None if x is None else x.copy() for x in A.D]
                N_A = [None if x is None else x.copy() for x in A.N]
                if N_A[sA] is None:
                    N_A[sA] = np.zeros(side_len(A, sideA), dtype=float)
                N_A[sA][i0:i0+mm] = -beta * flux_seg

                D_A[sA] = None                    # Neumann wins on A’s interface side
                A.solver.updateBC(D_A, N_A)
                uA, dirA = A.solver.solve(True)
                A.u = uA

                # 3) update the Dirichlet guess using A's temperature trace (relaxation)
                new_seg = np.array(dirA[sA][i0:i0+mm], dtype=float)
                diff = new_seg - dirichlet_guess[key][:mm]
                if diff.size:
                    max_res = max(max_res, float(np.max(np.abs(diff))))
                dirichlet_guess[key][:mm] = omega * new_seg + (1.0 - omega) * dirichlet_guess[key][:mm]

            if verbose:
                print(f"[DN] iter {k+1:02d}: residual={max_res:.3e}")

            last_res = max_res
            if max_res < tol:
                if verbose:
                    print(f"[DN] Converged in {k+1} iterations (tol={tol})")
                return {"iterations": k + 1, "residual": float(last_res), "omega": float(omega), "tol": float(tol)}

        # didn’t meet tol but finished max_iter — still return a summary
        return {"iterations": max_iter, "residual": float(last_res), "omega": float(omega), "tol": float(tol)}
