from room import Room
from cli_parser import args
from cli_helpers import *


class Apartment:
    def __init__(self, layout="default", dx=1 / 20):
        """
        Build the apartment layout and define subdomain coupling.

        layout = "default":
            omega1 1x1 (top-left),
            omega2 1x2 (tall middle),
            omega3 1x1 (bottom-right).

        layout = "alternative":
            same omega1/omega2/omega3,
            plus omega4 0.5x0.5 attached above omega3 on the right side.
        """

        if layout == "alternative":
            # create rooms
            omega1 = Room("", dx=dx, shape=(1, 1), heater_sides=["left"], window_sides=[])
            if args.verbose:
                print(dim("Successfully created room 'Omega_1'"))

            omega2 = Room("", dx=dx, shape=(1, 2), heater_sides=["top"], window_sides=["bottom"])
            if args.verbose:
                print(dim("Successfully created room 'Omega_2'"))

            omega3 = Room("", dx=dx, shape=(1, 1), heater_sides=["right"], window_sides=[])
            if args.verbose:
                print(dim("Successfully created room 'Omega_3'"))

            omega4 = Room(
                "",
                dx=dx,
                shape=(1 / 2, 1 / 2),
                heater_sides=["bottom"],
                window_sides=[],
            )
            if args.verbose:
                print(dim("Successfully created room 'Omega_4'"))

            """
            alternative layout geometry (project3a):
              omega1: bottom-left  (1x1)
              omega2: tall middle  (1x2)
              omega3: top-right    (1x1)
              omega4: lower-right small (0.5x0.5), tucked below-left of omega3

            interfaces (in physical y-coordinates of omega2, height=2.0):
              - omega1 ↔ omega2.left  on y ∈ [0.0, 1.0]   (bottom half)
              - omega3 ↔ omega2.right on y ∈ [1.0, 2.0]   (top half)
              - omega4 ↔ omega2.right on y ∈ [0.5, 1.0]   (middle band)

            and:
              - omega4.top touches the LEFT HALF of omega3.bottom.
            """

            # omega1 <-> omega2
            coupling1_2 = {
                "neighbor": omega2,
                "side": "right",
                "start": 0.0,
                "end": 1.0,
            }
            coupling2_1 = {
                "neighbor": omega1,
                "side": "left",
                "start": 0.0,
                "end": 1.0,
                "type": "dirichlet",
            }

            # omega2 <-> omega3 (top half of omega2.right)
            coupling2_3 = {
                "neighbor": omega3,
                "side": "right",
                "start": 1.0,
                "end": 2.0,
                "type": "dirichlet",
            }
            coupling3_2 = {
                "neighbor": omega2,
                "side": "left",
                "start": 0.0,
                "end": 1.0,
            }

            # omega2 <-> omega4 (middle band of omega2.right, y in [0.5,1.0])
            coupling2_4 = {
                "neighbor": omega4,
                "side": "right",
                "start": 0.5,
                "end": 1.0,
                "type": "dirichlet",
            }
            coupling4_2 = {
                "neighbor": omega2,
                "side": "left",
                "start": 0.0,
                "end": 0.5,
                # default (no "type") => Neumann for omega4
            }

            # omega3 <-> omega4 (omega3.bottom left half ↔ omega4.top full)
            coupling3_4 = {
                "neighbor": omega4,
                "side": "bottom",
                "start": 0.0,
                "end": 0.5,
                # Neumann for omega3
            }
            coupling4_3 = {
                "neighbor": omega3,
                "side": "top",
                "start": 0.0,
                "end": 1.0,
                "type": "dirichlet",
            }

            # attach couplings
            omega1.add_coupling(coupling1_2)
            omega2.add_coupling(coupling2_1)
            omega2.add_coupling(coupling2_3)
            omega2.add_coupling(coupling2_4)
            omega3.add_coupling(coupling3_2)
            omega3.add_coupling(coupling3_4)
            omega4.add_coupling(coupling4_2)
            omega4.add_coupling(coupling4_3)

            if args.verbose:
                print(dim("Successfully coupled all rooms (alternative)"))
            self.rooms = [omega1, omega2, omega3, omega4]
            if args.verbose:
                self.names = {
                    omega1: "Omega_1",
                    omega2: "Omega_2",
                    omega3: "Omega_3",
                    omega4: "Omega_4",
                }


        elif layout == "default":
            # --- create rooms ------------------------------------------------
            omega1 = Room("", dx, shape=(1, 1), heater_sides=["left"], window_sides=[])
            if args.verbose:
                print(dim("Successfully created room 'Omega_1'"))

            omega2 = Room("", dx, (1, 2), heater_sides=["top"], window_sides=["bottom"])
            if args.verbose:
                print(dim("Successfully created room 'Omega_2'"))

            omega3 = Room("", dx, shape=(1, 1), heater_sides=["right"], window_sides=[])
            if args.verbose:
                print(dim("Successfully created room 'Omega_3'"))

            """
            GEOMETRY for 'default':

            omega1 (1x1) touches TOP HALF of omega2.left
            omega3 (1x1) touches BOTTOM HALF of omega2.right
            omega2 is tall (1x2)

            So:
              omega2.left  with omega1 -> y in [1.0, 2.0]
              omega2.right with omega3 -> y in [0.0, 1.0]
            """

            # omega1 ↔ omega2
            omega1.add_coupling({
                "neighbor": omega2,
                "side": "right",
                "start": 0.0,
                "end": 1.0,
            })
            omega2.add_coupling({
                "neighbor": omega1,
                "side": "left",
                "start": 1.0,
                "end": 2.0,
                "type": "dirichlet",
            })

            # omega2 ↔ omega3
            omega2.add_coupling({
                "neighbor": omega3,
                "side": "right",
                "start": 0.0,
                "end": 1.0,
                "type": "dirichlet",
            })
            omega3.add_coupling({
                "neighbor": omega2,
                "side": "left",
                "start": 0.0,
                "end": 1.0,
            })

            if args.verbose:
                print(dim("Successfully coupled all rooms (default)"))

            self.rooms = [omega1, omega2, omega3]

            if args.verbose:
                self.names = {}
                for i in range(3):
                    self.names[self.rooms[i]] = f"Omega_{i+1}"
        else:
            raise NotImplementedError("only default and alternative layouts implemented")

    def iterate(self):
        # Dirichlet-Neumann style sweep:
        # 1. solve omega2 first (the tall middle room),
        # 2. then solve the neighbors using the flux (neumann).
        if args.geometry == "default":
            # rooms = [omega1, omega2, omega3]
            omega1, omega2, omega3 = self.rooms

            # step 1: middle room with Dirichlet from neighbors
            omega2.iterate_room()
            # step 2: outer rooms with Neumann from middle
            omega1.iterate_room()
            omega3.iterate_room()
            if args.verbose:
                for r in self.rooms:
                    print(dim(f"    Solved temperatures in room '{self.names[r]}'"))
            return

        elif args.geometry == "alternative":
            # rooms = [omega1, omega2, omega3, omega4]
            omega1, omega2, omega3, omega4 = self.rooms

            # step 1: middle room first
            omega2.iterate_room()
            # step 2: others
            omega1.iterate_room()
            omega3.iterate_room()
            omega4.iterate_room()
            if args.verbose:
                for r in self.rooms:
                    print(dim(f"    Solved temperatures in room '{self.names[r]}'"))
            return

        raise NotImplementedError("geometry problem in iterate()")

    def plot(self):
        import matplotlib.pyplot as plt
        import numpy as np

        if args.geometry == "default":
            # --- your existing default plotting (unchanged) ---
            Nxs = [room.Nx for room in self.rooms]
            Nys = [room.Ny for room in self.rooms]
            X = sum(Nxs) - 2
            Y = max(Nys)

            array = np.zeros((Y, X))
            array[: Nys[0], : Nxs[0]] += self.rooms[0].u.reshape((Nys[0], Nxs[0]))
            array[:, Nxs[0] - 1 : Nxs[0] + Nxs[1] - 1] += self.rooms[1].u.reshape((Nys[1], Nxs[1]))
            array[Y - Nys[2] :, Nxs[0] - 2 + Nxs[1] : Nxs[0] - 2 + Nxs[1] + Nxs[2]] += \
                self.rooms[2].u.reshape((Nys[2], Nxs[2]))

            array[: Nys[0], Nxs[0] - 1] /= 2
            array[Y - Nys[2] :, Nxs[0] + Nxs[1] - 2] /= 2

            eps = args.dx / 2
            plt.imshow(
                array,
                aspect=1,
                origin="lower",
                extent=[
                    -eps,
                    sum(room.Lx for room in self.rooms) + eps,
                    -eps,
                    eps + max([room.Ly for room in self.rooms]),
                ],
                cmap="seismic",
            )
            plt.colorbar()
            plt.show()
            return

        elif args.geometry == "alternative":
            omega1, omega2, omega3, omega4 = self.rooms

            dx = args.dx

            Lx_total = 3.0
            Ly_total = 2.0

            Nx_global = int(round(Lx_total / dx)) + 1   # columns
            Ny_global = int(round(Ly_total / dx)) + 1   # rows

            global_array = np.full((Ny_global, Nx_global), np.nan)

            #helper to place a room field u into global_array using physical extents
            def place_room(u_vec, room, x0, x1, y0, y1, average=True):
                u_local = u_vec.reshape((room.Ny, room.Nx))  # (Ny rows, Nx cols)


                ix0 = int(round(x0 / dx))
                ix1 = int(round(x1 / dx))  #inclusive end in physical is exclusive in slicing

                iy0 = int(round(y0 / dx))
                iy1 = int(round(y1 / dx))

                sub_Nx = ix1 - ix0 + 1
                sub_Ny = iy1 - iy0 + 1
                tgt_x_slice = slice(ix0, ix0 + room.Nx)
                tgt_y_slice = slice(iy0, iy0 + room.Ny)

                existing = global_array[tgt_y_slice, tgt_x_slice]

                if average and np.any(~np.isnan(existing)):

                    mask_new = np.isnan(existing)   #where there's already data, average

                    existing[mask_new] = u_local[mask_new]  #fill untouched spots first

                    both_mask = ~mask_new        # average where both exist
                    existing[both_mask] = 0.5 * (existing[both_mask] + u_local[both_mask])
                    global_array[tgt_y_slice, tgt_x_slice] = existing
                else:
                    global_array[tgt_y_slice, tgt_x_slice] = u_local
            #Ω1
            place_room(
                omega1.u,
                omega1,
                x0=0.0, x1=1.0,
                y0=0.0, y1=1.0,
            )

            #Ω2:
            place_room(
                omega2.u,
                omega2,
                x0=1.0, x1=2.0,
                y0=0.0, y1=2.0,
            )

            #Ω3:
            place_room(
                omega3.u,
                omega3,
                x0=2.0, x1=3.0,
                y0=1.0, y1=2.0,
            )

            #Ω4:
            place_room(
                omega4.u,
                omega4,
                x0=2.0, x1=2.5,
                y0=0.5, y1=1.0,
            )
            nan_mask = np.isnan(global_array)
            if np.any(nan_mask):
                global_array[nan_mask] = 15.0  # neutral interior-ish temp

            plt.imshow(
                global_array,
                origin="lower",
                aspect="equal",
                extent=[0.0, Lx_total, 0.0, Ly_total],
                cmap="seismic",
            )
            plt.colorbar()
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            plt.title("Temperature field (alternative layout)")
            plt.show()

            return
