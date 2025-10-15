import numpy as np
from heatSolver import heatSolver

"""
-------------------------------------------------------------
READ ME FIRST
-------------------------------------------------------------
This script builds the geometry and BCs for the apartment of the project.

Each room (Ω1, Ω2, Ω3) gets its own solver.

What you need to know:

1️. You can set which walls are HEATERS or WINDOWS 
 by editing the dictionaries when you call the function:

        heater_walls = {
            "omega1": ["left"],       #heater on left wall of room 1
            "omega3": ["right"]       #heater on right wall of room 3
        }

        window_walls = {
            "omega2": ["bottom"]      #window on bottom wall of room 2
        }

     Valid side names: "bottom", "left", "top", "right" 
     (type them exactly right — or the code will yell at you)

2. Call the main function like this:
        rooms, layout = build_room_solvers(dx, heater_walls, window_walls)

     - dx → grid spacing (like 1/20)
     - rooms → dictionary with one solver per room
     - layout → small helper dictionary that says how rooms connect

3. Each solver is stored like this:
        rooms["omega1"]["solver"]  → solver object for room 1
        rooms["omega2"]["solver"]  → solver object for room 2
        rooms["omega3"]["solver"]  → solver object for room 3

4. layout["inner1"] and layout["inner2"] tell you 
     which parts of Ω2’s walls connect to the other rooms.
     You’ll need these for the Dirichlet–Neumann iteration.

5. This file doesn’t solve the equations yet — 
     it just builds the “apartment” setup.
     You’ll use this in the main script that runs the iteration.
-------------------------------------------------------------
"""

def uniform_list(val, n):   #inputs: the value you want to repeat and how many times you want to repeat it
    return [float(val) for _ in range(n)]   #output is list

def build_room_solvers(dx, heater_walls=None, window_walls=None):
    heater_walls = heater_walls or {}
    window_walls = window_walls or {}
    normal_wall_temp = 15  #default wall temperature
    heater_temp = 40      #temperature of heaters
    window_temp = 5      #temperature of windows

    #Room sizes (Lx, Ly)
    L1 = (1.0, 1.0)
    L2 = (1.0, 2.0)
    L3 = (1.0, 1.0)
    L4 = (0.5, 0.5)

    #Function to build the wall arrays for Dirichlet/Neumann
    def makeBCs(Lx, Ly, heater_sides, window_sides):

        #in case the user types mistakes
        valid_sides = {"bottom", "left", "top", "right"}
        groups_to_check = [
            ("heater_sides", heater_sides),
            ("window_sides", window_sides)
        ]

        for group_name, sides in groups_to_check:
            #Check each side name given by the user
            for s in sides:
                if s not in valid_sides:
                    raise ValueError(
                        f"Be careful! You wrote jibberish in '{s}' under {group_name}. "
                        f"Valid names are: {', '.join(sorted(valid_sides))}."
                    )

        Nx = int(Lx / dx) + 1 #number of grid points in x direction
        Ny = int(Ly / dx) + 1 #number of grid points in y direction

        D_bottom = uniform_list(normal_wall_temp, Nx) #we set the normal wall as the default value
        D_left   = uniform_list(normal_wall_temp, Ny)
        D_top    = uniform_list(normal_wall_temp, Nx)
        D_right  = uniform_list(normal_wall_temp, Ny)

        #Apply window=5 and heater=40 overrides
        #window_sides is a list of strings that tells us which walls have windows
        #heater_sides is a list of strings that tells us which walls have heaters
        for s in window_sides:
            if   s == "bottom": D_bottom = uniform_list(window_temp, Nx)
            elif s == "left":   D_left   = uniform_list(window_temp, Ny)
            elif s == "top":    D_top    = uniform_list(window_temp, Nx)
            elif s == "right":  D_right  = uniform_list(window_temp, Ny)
        for s in heater_sides:
            if   s == "bottom": D_bottom = uniform_list(heater_temp, Nx)
            elif s == "left":   D_left   = uniform_list(heater_temp, Ny)
            elif s == "top":    D_top    = uniform_list(heater_temp, Nx)
            elif s == "right":  D_right  = uniform_list(heater_temp, Ny)

        #Do not use uniform_list yet! Otherwise you set the flux to zero
        N_bottom = None
        N_left   = None
        N_top    = None
        N_right  = None

        return (D_bottom, D_left, D_top, D_right), (N_bottom, N_left, N_top, N_right), Nx, Ny

    
    rooms = {}

    #Room Ω1
    hot1 = set(heater_walls.get("omega1", []))
    win1 = set(window_walls.get("omega1", []))
    D1, N1, Nx1, Ny1 = makeBCs(*L1, hot1, win1)
    # heatSolver(dx,sides,dirichletBC,neumanBC)
    solver1 = heatSolver(dx, L1, D1, N1)
    rooms["omega1"] = {"solver": solver1, "Nx": Nx1, "Ny": Ny1}

    #Room Ω2
    hot2 = set(heater_walls.get("omega2", []))
    win2 = set(window_walls.get("omega2", []))
    D2, N2, Nx2, Ny2 = makeBCs(*L2, hot2, win2)
    solver2 = heatSolver(dx, L2, D2, N2)
    rooms["omega2"] = {"solver": solver2, "Nx": Nx2, "Ny": Ny2}

    #Room Ω3
    hot3 = set(heater_walls.get("omega3", []))
    win3 = set(window_walls.get("omega3", []))
    D3, N3, Nx3, Ny3 = makeBCs(*L3, hot3, win3)
    solver3 = heatSolver(dx, L3, D3, N3)
    rooms["omega3"] = {"solver": solver3, "Nx": Nx3, "Ny": Ny3}

    #Room Ω4
    hot4 = set(heater_walls.get("omega4", []))
    win4 = set(window_walls.get("omega4", []))
    D4, N4, Nx4, Ny4 = makeBCs(*L4, hot4, win4)
    solver4 = heatSolver(dx, L4, D4, N4)
    rooms["omega4"] = {"solver": solver4, "Nx": Nx4, "Ny": Ny4}

    #layout is a dictionary that describes hw the rooms connect
    layout = {
        "Ny1": Ny1,
        "Ny2": Ny2,
        "Ny3": Ny3,
        "Inner1": slice(0, Ny1),
        "Inner2": slice(Ny2 - Ny3, Ny2)
    }
    return rooms, layout
