from room import Room
from cli_parser import args
from cli_helpers import *


class Apartment:
    def __init__(self, layout="default", dx=1 / 20):
        """
        Build the apartment geometry and define how rooms are coupled.

        layout = "default":
            Omega1 (1x1) on top-left
            Omega2 (1x2) tall in the middle
            Omega3 (1x1) bottom-right

            Geometry picture (y up):
                Omega1 | (top half of Omega2)
                       |
                -------+---------
                       | (bottom half of Omega2) | Omega3

            Important: Omega1 only touches the TOP HALF of Omega2's left wall.
                       Omega3 only touches the BOTTOM HALF of Omega2's right wall.

        layout = "alternative":
            Omega1, Omega2, Omega3 like above + Omega4 (1/2 x 1/2 "small room").
            This is the extended problem (3a).
        """

        if layout == "alternative":
            # --- Create rooms -------------------------------------------------
            omega1 = Room(
                "",
                dx=dx,
                shape=(1, 1),
                heater_sides=["left"],
                window_sides=[],
            )
            if args.verbose:
                print(dim("Successfully created room 'Omega_1'"))

            omega2 = Room(
                "",
                dx=dx,
                shape=(1, 2),
                heater_sides=["top"],
                window_sides=["bottom"],
            )
            if args.verbose:
                print(dim("Successfully created room 'Omega_2'"))

            omega3 = Room(
                "",
                dx=dx,
                shape=(1, 1),
                heater_sides=["right"],
                window_sides=[],
            )
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

            # --- Couplings for the "alternative" layout ----------------------
            #NOTE: we haven't audited these segments yet in detail.
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

            coupling2_4 = {
                "neighbor": omega4,
                "side": "right",
                "start": 0.5,
                "end": 1.0,
                "type": "dirichlet",
            }
            coupling4_2 = {
                "neighbor": omega2,
                "side": "right",
                "start": 0.0,
                "end": 0.5,
            }

            coupling3_4 = {
                "neighbor": omega4,
                "side": "bottom",
                "start": 0.0,
                "end": 0.5,
                "type":"dirichlet"
            }
            coupling4_3 = {
                "neighbor": omega3,
                "side": "top",
                "start": 0.0,
                "end": 0.5,
            }

            # Attach couplings to rooms
            omega1.add_coupling(coupling1_2)
            omega2.add_coupling(coupling2_1)
            omega2.add_coupling(coupling2_3)
            omega2.add_coupling(coupling2_4)
            omega3.add_coupling(coupling3_2)
            omega3.add_coupling(coupling3_4)
            omega4.add_coupling(coupling4_2)
            omega4.add_coupling(coupling4_3)

            if args.verbose:
                print(dim("Successfully coupled all rooms"))

            #Register rooms in order
            self.rooms = [omega1, omega2, omega3, omega4]

            if args.verbose:
                self.names = {}
                for i in range(4):
                    self.names[self.rooms[i]] = f"Omega_{i+1}"

        elif layout == "default":
            # --- Create rooms ----------------------------
            omega1 = Room(
                "",
                dx,
                shape=(1, 1),
                heater_sides=["left"],
                window_sides=[],
            )
            if args.verbose:
                print(dim("Successfully created room 'Omega_1'"))

            omega2 = Room(
                "",
                dx,
                (1, 2),
                heater_sides=["top"],
                window_sides=["bottom"],
            )
            if args.verbose:
                print(dim("Successfully created room 'Omega_2'"))

            omega3 = Room(
                "",
                dx,
                shape=(1, 1),
                heater_sides=["right"],
                window_sides=[],
            )
            if args.verbose:
                print(dim("Successfully created room 'Omega_3'"))

            # --- Couplings for the "default" layout --------------------------
            """
                GEOMETRY / INTERFACE MAP (very important)
                Room sizes:
                    omega1: 1 x 1
                    omega2: 1 x 2   (tall, spans both top and bottom)
                    omega3: 1 x 1

                Contact logic:
                    - omega1 only touches the TOP HALF of omega2's LEFT wall
                    - omega3 only touches the BOTTOM HALF of omega2's RIGHT wall

                Coordinate convention for start/end:
                    For vertical walls ("left" / "right"), start/end are measured
                    from bottom→top along that wall in *physical length units*.

                    omega2 has total height Ly = 2.0, so:
                        y in [0.0, 1.0] = bottom half of omega2
                        y in [1.0, 2.0] = top half of omega2

                Bug we had:
                    The omega1–omega2 interface looked wrong (discontinuous),
                    while omega2–omega3 looked ok. That happened because we were
                    assigning the wrong half of omega2's wall to the wrong neighbor.

                Correct coupling (what we enforce below):
                    - omega2.left  with omega1 -> use [1.0, 2.0]  (TOP half)
                    - omega2.right with omega3 -> use [0.0, 1.0]  (BOTTOM half)
                """
            omega1.add_coupling(
                {
                    "neighbor": omega2,
                    "side": "right",
                    "start": 0.0,
                    "end": 1.0,
                }
            )
            omega2.add_coupling(
                {
                    "neighbor": omega1,
                    "side": "left",
                    "start": 0.0,  # top half of omega2: y ∈ [1.0, 2.0]
                    "end": 1.0,
                    "type": "dirichlet",
                }
            )

            # omega2 → omega3
            # omega3 sits against the BOTTOM HALF of omega2's right wall
            omega2.add_coupling(
                {
                    "neighbor": omega3,
                    "side": "right",
                    "start": 1.0,  # bottom half of omega2: y ∈ [0.0, 1.0]
                    "end": 2.0,
                    "type": "dirichlet",
                }
            )

            # omega3 → omega2
            omega3.add_coupling(
                {
                    "neighbor": omega2,
                    "side": "left",
                    "start": 0.0,
                    "end": 1.0,
                }
            )

            if args.verbose:
                print(dim("Successfully coupled all rooms"))

            # Register rooms in order
            self.rooms = [omega1, omega2, omega3]

            if args.verbose:
                self.names = {}
                for i in range(3):
                    self.names[self.rooms[i]] = f"Omega_{i+1}"

        else:
            raise NotImplementedError("only default and alternative layouts implemented")

    def iterate(self):
        """
        Perform one Dirichlet–Neumann iteration sweep on all rooms.

        NOTE:
        - The 'dirichlet' flag in the zip(...) arrays isn't actually
          used inside iterate_room() in this file, but we're keeping
          the structure the team had so we don't break anything.
        """
        if args.geometry == "default":
            for room, dirichlet in zip(self.rooms, [True, False, True]):
                room.iterate_room()
                if args.verbose:
                    print(dim(f"    Solved temperatures in room '{self.names[room]}'"))
            return

        elif args.geometry == "alternative":
            for room, dirichlet in zip(self.rooms, [True, False, True, True]):
                room.iterate_room()
                if args.verbose:
                    print(dim(f"    Solved temperatures in room '{self.names[room]}'"))
            return

        raise NotImplementedError("geometry problem in iterate()")

    def plot(self):
        """
        Stitch subdomain solutions together into one array and show temperature.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if args.geometry == "default":
            #Grab grid sizes from each room
            Nxs = [room.Nx for room in self.rooms]
            Nys = [room.Ny for room in self.rooms]

            #Combined array size:
            #X direction: sum of widths minus overlap columns
            #Y direction: max height
            X = sum(Nxs) - 2
            Y = max(Nys)

            array = np.zeros((Y, X))

            #Room 1 (omega1) goes top-left
            array[: Nys[0], : Nxs[0]] += self.rooms[0].u.reshape((Nys[0], Nxs[0]))

            #Room 2 (omega2) spans the full height in the middle
            array[:, Nxs[0] - 1 : Nxs[0] + Nxs[1] - 1] += self.rooms[1].u.reshape(
                (Nys[1], Nxs[1])
            )

            #Room 3 (omega3) is bottom-right
            array[
                Y - Nys[2] :,
                Nxs[0] - 2 + Nxs[1] : Nxs[0] - 2 + Nxs[1] + Nxs[2],
            ] += self.rooms[2].u.reshape((Nys[2], Nxs[2]))

            #Average values on the shared interface columns to avoid double-count stripe
            array[: Nys[0], Nxs[0] - 1] /= 2
            array[Y - Nys[2] :, Nxs[0] + Nxs[1] - 2] /= 2

            #Plot extent in physical coordinates
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
                cmap="viridis",  # teacher used 'seismic' to spot jumps
            )
            plt.colorbar()
            plt.show()
            return

        elif args.geometry == "alternative":
            Nxs = [room.Nx for room in self.rooms]
            Nys = [room.Ny for room in self.rooms]

            #For alternative layout, last room (omega4) isn't just glued in a line,
            # so X is sum of first 3 widths, Y is tallest height.
            X = sum(Nxs[:-1])
            Y = max(Nys)

            array = np.zeros((Y, X))

            # omega1 top-left
            array[: Nys[0], : Nxs[0]] += self.rooms[0].u.reshape((Nys[0], Nxs[0]))

            #omega2 middle column
            array[:, Nxs[0] : Nxs[0] + Nxs[1]] += self.rooms[1].u.reshape(
                (Nys[1], Nxs[1])
            )

            #omega3 bottom-right
            array[
                Y - Nys[2] :,
                Nxs[0] + Nxs[1] : Nxs[0] + Nxs[1] + Nxs[2],
            ] += self.rooms[2].u.reshape((Nys[2], Nxs[2]))

            #omega4 (the little extra room)
            array[
                Y - Nys[2] - Nys[3] : Y - Nys[2],
                Nxs[0] + Nxs[1] : Nxs[0] + Nxs[1] + Nxs[3],
            ] += self.rooms[3].u.reshape((Nys[3], Nxs[3]))

            plt.imshow(array, aspect=1, origin="lower", cmap="seismic")
            plt.colorbar()
            plt.show()
            return
