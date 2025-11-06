import numpy as np
from heatSolver import HeatSolver

''' This is the way the axes for the coupling ("start" and "end") will be defined:
                                                                  @                    
                                                                  @@@@                
           @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@            
                                                                   @@@@@              
                                                                  @@   
   @      @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@      @    
  @@@     @@                                                             @@     @@@   
 @@@@@    @@                                                             @@     @@@@  
@@@@@@@   @@                                                             @@    @@@@@@ 
@@ @@@@@  @@                                                             @@   @@ @@ @@
   @@     @@                                                             @@      @@   
   @@     @@                                                             @@      @@   
   @@     @@                                                             @@      @@   
   @@     @@                                                             @@      @@   
   @@     @@                                                             @@      @@   
   @@     @@                                                             @@      @@   
   @@     @@                                                             @@      @@   
   @@     @@                                                             @@      @@   
   @@     @@                                                             @@      @@   
   @@     @@                                                             @@      @@   
   @@     @@                                                             @@      @@   
   @@     @@                                                             @@      @@   
   @@     @@                                                             @@      @@   
   @@     @@                                                             @@      @@   
   @@     @@                                                             @@      @@   
   @@     @@                                                             @@      @@   
   @@     @@                                                             @@      @@   
   @@     @@                                                             @@      @@   
   @@     @@                                                             @@      @@   
   @@     @@                                                             @@      @@   
   @@     @@                                                             @@      @@   
   @@     @@                                                             @@      @@   
   @@     @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@      @@   
                                                                  @                   
                                                                  @@@@                
           @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@            
                                                                   @@@@@              
                                                                  @@                  '''


class Room:

    walls_order = {"bottom": 0, "left": 1, "top": 2, "right": 3}

    @staticmethod
    def opposite_side(side):
        opposites = {
            "bottom": "top",
            "top": "bottom",
            "left": "right",
            "right": "left"
        }
        return opposites.get(side, None)


    '''
    A class to represent a room in the apartment.
    
    Attributes
    ----------
        dx : float
            The grid spacing.
        shape : tuple
            The dimensions of the room (Lx, Ly).
        heater_sides : list
            The sides of the room with heaters (e.g., ["left", "right"]).
        window_sides : list
            The sides of the room with windows (e.g., ["top"]).
        heater_temp : float
            The temperature of the heaters (default is 40).
        window_temp : float
            The temperature of the windows (default is 5).
        normal_wall_temp : float
            The temperature of the normal walls (default is 15).
        side_to_indices : dict
            A mapping from side names to their corresponding grid indices.
        neighbors : list
            A list of neighboring room couplings.
        couplings : dict
            A dictionary to store coupling details by neighbor name.
        u : numpy.ndarray
            The temperature distribution in the room.
        D : list
            Dirichlet boundary conditions for each side [bottom, left, top, right].
        N : list
            Neumann boundary conditions for each side [bottom, left, top, right].
        solver : heatSolver.heatSolver
            The heat solver instance for the room.
    Methods
    -------
        add_coupling(coupling)
            Add a coupling to a neighboring room.
        get_boundary_value(side, start, end)
            Get the boundary value for a specific side and start index.
        iterate_room()
            Update the room's temperature distribution.
    '''

    def __init__(self, aname, dx, shape, relaxation = 0.8, heater_sides=None, window_sides=None, heater_temp=40, window_temp=5, normal_wall_temp=15):
        self.aname = aname
        self.relaxation = relaxation
        self.dx = dx
        self.Lx, self.Ly = shape
        self.Nx = int(self.Lx / self.dx) + 1 #number of grid points in x direction
        self.Ny = int(self.Ly / self.dx) + 1 #number of grid points in y direction
        self.N_tot = self.Nx * self.Ny
        self.heater_sides = heater_sides or []
        self.window_sides = window_sides or []
        self.heater_temp = heater_temp
        self.window_temp = window_temp
        self.normal_wall_temp = normal_wall_temp
        self.side_to_indices = {}
        self.neighbors = []  # List to store neighboring room couplings 
        self.couplings = {}  # Dictionary to store coupling details by neighbor name
        self.u = np.zeros(self.N_tot)  # Initialize temperature array
        self.new_u = np.zeros(self.N_tot)  # For relaxation
        self.D = [None, None, None, None]  # Dirichlet BCs: [bottom, left, top, right]
        self.N = [None, None, None, None]  # Neumann BCs: [bottom, left, top, right]
        self.global_boundary_mask = {"bottom": None, "left": None, "top": None, "right": None}
        self.boundary_information = {"Dirichlet": self.D.copy(), "Neumann": self.N.copy()}

        self._initialize_BCs(self.heater_sides, self.window_sides)
        
        self.solver = HeatSolver(self.dx, (self.Lx, self.Ly), self.D, self.N)

    
    def _initialize_BCs(self, heater_sides, window_sides):
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
                        f"Be careful! You wrote gibberish in '{s}' under {group_name}. "
                        f"Valid names are: {', '.join(sorted(valid_sides))}."
                    )

        #Boundary Indices
        self.side_to_indices["bottom"] = np.arange(0, self.Nx)
        self.side_to_indices["left"] = np.arange(0, self.N_tot, self.Nx)
        self.side_to_indices["top"] = np.arange(self.N_tot - self.Nx, self.N_tot)
        self.side_to_indices["right"] = np.arange(self.Nx - 1, self.N_tot, self.Nx)

        self.D[0] = np.full(self.Nx, self.normal_wall_temp) #we set the normal wall as the default value
        self.D[1] = np.full(self.Ny, self.normal_wall_temp)
        self.D[2] = np.full(self.Nx, self.normal_wall_temp)
        self.D[3] = np.full(self.Ny, self.normal_wall_temp)

        #Apply window=5 and heater=40 overrides
        #window_sides is a list of strings that tells us which walls have windows
        #heater_sides is a list of strings that tells us which walls have heaters
        for s in window_sides:
            if   s == "bottom": self.D[0] = np.full(self.Nx, self.window_temp)
            elif s == "left":   self.D[1] = np.full(self.Ny, self.window_temp)
            elif s == "top":    self.D[2] = np.full(self.Nx, self.window_temp)
            elif s == "right":  self.D[3] = np.full(self.Ny, self.window_temp)
        for s in heater_sides:
            if   s == "bottom": self.D[0] = np.full(self.Nx, self.heater_temp)
            elif s == "left":   self.D[1] = np.full(self.Ny, self.heater_temp)
            elif s == "top":    self.D[2] = np.full(self.Nx, self.heater_temp)
            elif s == "right":  self.D[3] = np.full(self.Ny, self.heater_temp)


    def get_boundary_indices(self, side, start, end, flat=True):
        full_boundary = self.side_to_indices[side]
        full_length = self.Lx if side in {"bottom", "top"} else self.Ly
        n = len(full_boundary)
        if flat:
            return np.zeros(n)
        x = np.linspace(0, full_length, n)
        mask = (x >= start) & (x <= end)
        return full_boundary[mask]

    def add_coupling(self, coupling):
        check_keys = {"neighbor", "side", "start", "end", "type"}
        if not isinstance(coupling, dict):
            raise ValueError(f"Coupling must be a dictionary with keys: {', '.join(sorted(check_keys))}.")
        if not isinstance(coupling["neighbor"], Room):
            raise ValueError("The 'neighbor' must be an instance of the Room class.")
        if coupling["side"] not in {"bottom", "left", "top", "right"}:
            raise ValueError("The 'side' must be one of: bottom, left, top, right.")
        if not isinstance(coupling["start"], float) or coupling["start"] < 0:
            raise ValueError("The 'start' must be a non-negative float.")
        if "end" in coupling and (not isinstance(coupling["end"], float) or coupling["end"] <= coupling["start"]):
            raise ValueError("The 'end' must be a float greater than 'start'.")
        if not "type" in coupling:
            coupling["type"] = "neumann"  # Default to Neumann if not specified  
        elif "type" in coupling and coupling["type"] not in {"dirichlet", "neumann"}:
            raise ValueError("The 'type' must be either 'dirichlet' or 'neumann'.")
        
        side = coupling["side"]
        start = coupling["start"]
        end = coupling["end"]
        bcType = coupling["type"]

        # Adjust BC to Neumann for the coupled side
        if bcType == "neumann":
            self.D[Room.walls_order[coupling["side"]]] = None  # Remove Dirichlet BC for this side always
        

        self.solver.updateBC(self.D, self.N)
        self.neighbors.append(coupling)
        self.generate_global_boundary_masks()

    def get_boundary_value(self, side, start, end):
        '''Get the boundary value for a specific side and start index.
        Parameters
        ----------
            side : str
                The side of the room ("bottom", "left", "top", "right").
            start : int
                The starting index along the specified side.
            end : int
                The ending index along the specified side.
        Returns
        -------
            numpy.ndarray or None
                The boundary value if set, otherwise None.'''
        valid_sides = {"bottom", "left", "top", "right"}

        if side not in valid_sides:
            raise ValueError(f"Invalid side '{side}'. Valid sides are: {', '.join(sorted(valid_sides))}.")

        

        full_boundary = self.side_to_indices[side]
        full_length = self.Lx if side in {"bottom", "top"} else self.Ly
        n = len(full_boundary)

        x = np.linspace(0, full_length, n)

        mask = (x >= start) & (x <= end)

        selected_indices = full_boundary[mask]

        return np.array([self.u[i] for i in selected_indices])

    def give_border_start_and_end(self, room) -> tuple:
        my_start = my_end = None
        for coupling in self.neighbors:
            if coupling["neighbor"] == room:
                my_start = coupling["start"]
                my_end = coupling["end"]
                break
        if my_start is None or my_end is None:
            raise ValueError("The specified room is not a neighbor.")
        return my_start, my_end

    def generate_global_boundary_masks(self):
        for side in Room.walls_order.keys():
            n = self.Nx if side in {"bottom", "top"} else self.Ny
            mask = np.zeros(n, dtype=bool)     # <-- NumPy, not list
            for neighbor in self.neighbors:
                if side != neighbor["side"]:
                    continue
                full_length = self.Lx if side in {"bottom", "top"} else self.Ly
                x = np.linspace(0, full_length, n)
                mask |= (x >= neighbor["start"]) & (x <= neighbor["end"])
            self.global_boundary_mask[side] = mask

            print(f"Global boundary mask for side '{side}': {self.global_boundary_mask[side]}, room '{self.aname}' ")


    def create_full_boundary_array(self, values, side, start, end, coupling_type):
        full_boundary = self.side_to_indices[side]
        full_length = self.Lx if side in {"bottom", "top"} else self.Ly
        n = len(full_boundary)
        x = np.linspace(0, full_length, n)
        local_mask = [False if (x[i] < start) | (x[i] > end) else True for i in range(n)]
        full_boundary_array = np.zeros(n)
        
        j = 0
        for i in range(n):
            if local_mask[i]:
                full_boundary_array[i] = values[j]
                j += 1
            elif not self.global_boundary_mask[side][i]:
                full_boundary_array[i] = self.normal_wall_temp
            else:
                if coupling_type == "dirichlet" and self.D[Room.walls_order[side]] is not None:
                    full_boundary_array[i] = self.D[Room.walls_order[side]][i]
                elif coupling_type == "neumann" and self.N[Room.walls_order[side]] is not None:
                    full_boundary_array[i] = self.N[Room.walls_order[side]][i]
        return full_boundary_array

    def get_one_insideboundary_index(self, side):
        if side == "bottom":
            return self.side_to_indices[side] + self.Nx
        elif side == "top":
            return self.side_to_indices[side] - self.Nx
        elif side == "left":
            return self.side_to_indices[side] + 1
        elif side == "right":
            return self.side_to_indices[side] - 1
        
    def get_neumann_boundary_value(self, side):
        flux = None
        if side == "bottom":
            flux =  (self.u[self.side_to_indices[side]] - self.u[self.get_one_insideboundary_index(side)]) / self.dx
        elif side == "top":
            flux =  (self.u[self.side_to_indices[side]] - self.u[self.get_one_insideboundary_index(side)]) / self.dx
        elif side == "left":
            flux =  (self.u[self.side_to_indices[side]] - self.u[self.get_one_insideboundary_index(side)]) / self.dx
        elif side == "right":
            flux =  (self.u[self.side_to_indices[side]] - self.u[self.get_one_insideboundary_index(side)]) / self.dx
        if flux is None:
            raise ValueError(f"Invalid side '{side}' for Neumann boundary value.")
        flux = np.where(self.global_boundary_mask[side], flux, 0) # Set values outside the coupled region to 0
        return flux

    def generate_boundary_information(self):
        for side in Room.walls_order.keys():
            local_dirichlet = self.u[self.side_to_indices[side]]
            local_neumann = self.get_neumann_boundary_value(side)
            self.boundary_information["Dirichlet"][Room.walls_order[side]] = local_dirichlet[self.global_boundary_mask[side]] # ensures consistency with the full boundary
            self.boundary_information["Neumann"][Room.walls_order[side]] = local_neumann[self.global_boundary_mask[side]] 

    def get_boundary_information(self):
        return self.boundary_information

    def map_boundary_information(self, neighbor, coupling_type, side, start, end):
        neighbor_info = neighbor.get_boundary_information() # This contains entire boundary info of the neighbor
        if coupling_type == "dirichlet":
            values = neighbor_info["Dirichlet"][Room.walls_order[Room.opposite_side(side)]]
            full_boundary_array = self.create_full_boundary_array(values, side, start, end, coupling_type) # Here we create the full boundary array for our room, with only the coupled region filled with neighbor values
            self.D[Room.walls_order[side]] = full_boundary_array
        elif coupling_type == "neumann":
            values = neighbor_info["Neumann"][Room.walls_order[Room.opposite_side(side)]]
            full_boundary_array = self.create_full_boundary_array(values, side, start, end, coupling_type) # Here we create the full boundary array for our room, with only the coupled region filled with neighbor values
            self.N[Room.walls_order[side]] = full_boundary_array

    def iterate_room(self):
        '''Update the room's temperature distribution.'''
        #Update boundary conditions from neighbors
        for coupling in self.neighbors:
            neighbor = coupling["neighbor"]
            side = coupling["side"]
            start = coupling["start"]
            end = coupling["end"]
            coupling_type = coupling["type"]

            self.map_boundary_information(neighbor, coupling_type, side, start, end)


        self.solver.updateBC(self.D, self.N)

        self.new_u, _ = self.solver.solve()
        self.u = self.relaxation * self.new_u + (1 - self.relaxation) * self.u
        self.generate_boundary_information()

if __name__ == "__main__":
    four = False

    omega1 = Room("Omega 1", 0.01, (1.0, 1.0), heater_sides=["left"])
    omega2 = Room("Omega 2", 0.01, (1.0, 2.0), heater_sides=["top"], window_sides=["bottom"])
    omega3 = Room("Omega 3", 0.01, (1.0, 1.0), heater_sides=["right"])
    omega4 = Room("Omega 4", 0.01, (0.5, 0.5), heater_sides=["bottom"]) if four else None

    omega1.add_coupling({"neighbor": omega2, "side": "right", "start": 0.0, "end": 1.0, "type": "neumann"})
    omega2.add_coupling({"neighbor": omega1, "side": "left", "start": 0.0, "end": 1.0, "type": "dirichlet"})
    omega2.add_coupling({"neighbor": omega3, "side": "right", "start": 1.0, "end": 2.0, "type": "dirichlet"})
    omega2.add_coupling({"neighbor": omega4, "side": "right", "start": 0.5, "end": 1.0, "type": "dirichlet"}) if four else None
    omega3.add_coupling({"neighbor": omega2, "side": "left", "start": 0.0, "end": 1.0, "type": "neumann"})
    omega3.add_coupling({"neighbor": omega4, "side": "bottom", "start": 0.0, "end": 0.5, "type": "neumann"}) if four else None
    omega4.add_coupling({"neighbor": omega2, "side": "left", "start": 0.0, "end": 0.5, "type": "neumann"}) if four else None
    omega4.add_coupling({"neighbor": omega3, "side": "top", "start": 0.0, "end": 0.5, "type": "neumann"}) if four else None

    omega1.generate_boundary_information()
    omega2.generate_boundary_information()  
    omega3.generate_boundary_information()
    omega4.generate_boundary_information() if four else None

    for _ in range(10):
        omega2.iterate_room()
        omega1.iterate_room()
        omega4.iterate_room() if four else None
        omega3.iterate_room()
        
        

    print("Room 1 Temperature Distribution:\n", omega1.u.reshape((omega1.Ny, omega1.Nx)))
    print("Room 2 Temperature Distribution:\n", omega2.u.reshape((omega2.Ny, omega2.Nx)))
    print("Room 3 Temperature Distribution:\n", omega3.u.reshape((omega3.Ny, omega3.Nx)))
    print("Room 4 Temperature Distribution:\n", omega4.u.reshape((omega4.Ny, omega4.Nx))) if four else None

        # --- Combined temperature plot for all rooms ---
    import matplotlib.pyplot as plt
    import numpy as np

    # Domain and grid
    dx = omega1.dx
    Lx_tot = 3.0   # total width (Ω1 + Ω2 + Ω3)
    Ly_tot = 2.0   # total height
    Nx_tot = int(Lx_tot / dx) + 1
    Ny_tot = int(Ly_tot / dx) + 1

    # Global canvas (NaN = empty region -> white)
    global_T = np.full((Ny_tot, Nx_tot), np.nan)

    # Reshape each room’s field
    T1 = omega1.u.reshape(omega1.Ny, omega1.Nx)
    T2 = omega2.u.reshape(omega2.Ny, omega2.Nx)
    T3 = omega3.u.reshape(omega3.Ny, omega3.Nx)
    T4 = omega4.u.reshape(omega4.Ny, omega4.Nx) if four else None

    # --- Offsets (in grid points) ---
    # Ω1: bottom-left (1×1)
    x0_1, y0_1 = 0, 0
    # Ω2: middle (1×2)
    x0_2, y0_2 = omega1.Nx - 1, 0
    # Ω3: top-right (1×1)
    x0_3, y0_3 = omega1.Nx + omega2.Nx - 2, omega1.Ny - 1
    # Ω4: small 0.5×0.5 block shifted UP to y = 0.5 m
    x0_4 = x0_3
    y0_4 = int(0.5 / dx)  # 0.5 m offset upward

    # --- Insert room data into the global canvas ---
    global_T[y0_1:y0_1 + omega1.Ny, x0_1:x0_1 + omega1.Nx] = T1
    global_T[y0_2:y0_2 + omega2.Ny, x0_2:x0_2 + omega2.Nx] = T2
    global_T[y0_3:y0_3 + omega3.Ny, x0_3:x0_3 + omega3.Nx] = T3
    if four:
        global_T[y0_4:y0_4 + omega4.Ny, x0_4:x0_4 + omega4.Nx] = T4

    # --- Plot ---
    cmap = plt.cm.hot
    cmap.set_bad(color='white')  # white = no room region

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(
        global_T, cmap=cmap, origin='lower', aspect='equal',
        extent=[0, Lx_tot, 0, Ly_tot]
    )

    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Temperature (°C)")

    ax.set_title(f"Temperature Distribution (h={dx:.2f} m, DN coupling)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    plt.tight_layout()
    plt.show()

