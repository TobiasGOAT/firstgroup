from room import Room
from cli_parser import args
from cli_helpers import *


class Apartment:
    def __init__(self, layout="default", dx=1 / 20):
        if layout == "alternative":
            omega1 = Room(dx, (1, 1), ["left"], [])
            if args.verbose:
                print(dim("Successfully created room 'Omega_1'"))
            omega2 = Room(dx, (1, 2), ["top"], ["bottom"])
            if args.verbose:
                print(dim("Successfully created room 'Omega_2'"))
            omega3 = Room(dx, (1, 1), ["right"], [])
            if args.verbose:
                print(dim("Successfully created room 'Omega_3'"))
            omega4 = Room(dx, (1 / 2, 1 / 2), ["bottom"], [])
            if args.verbose:
                print(dim("Successfully created room 'Omega_4'"))
            coupling1_2 = {
                "neighbor": omega2,
                "side": "right",
                "start": 0.0,
                "end": 1.0,
            }
            coupling2_1 = {"neighbor": omega1, "side": "left", "start": 0.0, "end": 1.0}
            coupling2_3 = {
                "neighbor": omega3,
                "side": "right",
                "start": 1.0,
                "end": 2.0,
            }
            coupling3_2 = {"neighbor": omega2, "side": "left", "start": 0.0, "end": 1.0}
            coupling2_4 = {
                "neighbor": omega4,
                "side": "right",
                "start": 0.5,
                "end": 1.0,
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
            }
            coupling4_3 = {"neighbor": omega3, "side": "top", "start": 0.0, "end": 0.5}
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
            self.rooms = [omega1, omega2, omega3, omega4]
            if args.verbose:
                self.names={}
                for i in range(4):
                    self.names[self.rooms[i]]=f"Omega_{i+1}"
        elif layout == "default":
            omega1 = Room("",dx, shape=(1, 1), heater_sides=["left"], window_sides=[])
            if args.verbose:
                print(dim("Successfully created room 'Omega_1'"))
            omega2 = Room("",dx, (1, 2), heater_sides=["top"], window_sides=["bottom"])
            if args.verbose:
                print(dim("Successfully created room 'Omega_2'"))
            omega3 = Room("",dx, shape=(1, 1), heater_sides=["right"], window_sides=[])
            if args.verbose:
                print(dim("Successfully created room 'Omega_3'"))
            coupling1_2 = {
                "neighbor": omega2,
                "side": "right",
                "start": 0.0,
                "end": 1.0,
            }
            coupling2_1 = {"neighbor": omega1, "side": "left", "start": 0.0, "end": 1.0}
            coupling2_3 = {
                "neighbor": omega3,
                "side": "right",
                "start": 1.0,
                "end": 2.0,
            }
            coupling3_2 = {"neighbor": omega2, "side": "left", "start": 0.0, "end": 1.0}
            omega1.add_coupling(coupling1_2)
            omega2.add_coupling(coupling2_1)
            omega2.add_coupling(coupling2_3)
            omega3.add_coupling(coupling3_2)
            if args.verbose:
                print(dim("Successfully coupled all rooms"))
            self.rooms = [omega1, omega2, omega3]
            if args.verbose:
                self.names={}
                for i in range(3):
                    self.names[self.rooms[i]]=f"Omega_{i+1}"
        else:
            raise NotImplementedError("only default layout implemented")

    def iterate(self):
        for room in self.rooms:
            room.iterate_room()
            if args.verbose:
                print(dim(f"    Solved temperatures in room '{self.names[room]}'"))
    def plot(self):
        import matplotlib.pyplot as plt
        import numpy as np
        if args.geometry=="default":
            Nxs=[room.Nx for room in self.rooms]
            Nys=[room.Ny for room in self.rooms]
            X=sum(Nxs)
            Y=max(Nys)
            array=np.zeros((Y, X)) #advanced plotting going on here
            array[Y-Nys[0]:, :Nxs[0]]=self.rooms[0].u.reshape((Nys[0], Nxs[0]))
            array[:, Nxs[0]:Nxs[0]+Nxs[1]]=self.rooms[1].u.reshape((Nys[1], Nxs[1]))
            array[:Nys[2], Nxs[1]:Nxs[1]+Nxs[2]]=self.rooms[2].u.reshape((Nys[2], Nxs[2]))
            plt.imshow(array, aspect=1)
            plt.colorbar()
            plt.show()

