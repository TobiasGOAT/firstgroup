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
        self.boundary_mask = {"bottom": None, "left": None, "top": None, "right": None}

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

    def add_coupling(self, coupling):
        """
        Adds a coupling between this room and a neighboring room.

        Parameters
        coupling : dict
            A dictionary specifying the coupling details. Must contain the following keys:
            - "neighbor": An instance of the neighboring room (must be of type `room`).
            - "side": A string indicating which side of the NEIGHBOR room is coupled compared to THIS room ("bottom", "left", "top", "right").
            - "start": Length of the start along the specified side from THIS room's perspective.
            - "end": Length of the end along the specified side from THIS room's perspective.
            - "type": (Optional) Type of coupling, either "dirichlet" or "neumann". Default is "neumann".


        Raises
        ------
        ValueError
            If the coupling dictionary is missing required keys, contains invalid values, or types are incorrect.

        Example
        -------
        >>> coupling = {
        ...     "neighbor": room_instance,
        ...     "side": "left",
        ...     "start": 0.0,
        ...     "end": 5.0,
        ...     "type": "neumann"
        ... }
        >>> room.add_coupling(coupling)
        """
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
        
        # Adjust BC to Neumann for the coupled side
        if coupling["type"] == "neumann":
            full_side_length = self.Nx if coupling["side"] in {"bottom", "top"} else self.Ny
            self.N[Room.walls_order[coupling["side"]]] = np.zeros(full_side_length)
        self.D[Room.walls_order[coupling["side"]]] = None  # Remove Dirichlet BC for this side

        self.solver.updateBC(self.D, self.N,self.u)

        self.neighbors.append(coupling)

        self.generate_boundary_masks()

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

    def generate_boundary_masks(self):
        for side in Room.walls_order.keys():
            mask = [False] * (self.Nx if side in {"bottom", "top"} else self.Ny)
            for neighbor in self.neighbors:
                if side != neighbor["side"]:
                    continue
                full_boundary = self.side_to_indices[side]
                n = len(full_boundary)
                full_length = self.Lx if side in {"bottom", "top"} else self.Ly
                x = np.linspace(0, full_length, n)
                mask |= (x >= neighbor["start"]) & (x <= neighbor["end"])
            self.boundary_mask[side] = mask

    def create_full_boundary_array(self, values, side, start, end):
        full_boundary = self.side_to_indices[side]
        full_length = self.Lx if side in {"bottom", "top"} else self.Ly
        n = len(full_boundary)
        x = np.linspace(0, full_length, n)
        mask = [False if (x[i] < start) | (x[i] > end) else True for i in range(n)]
        full_boundary_array = np.zeros(n)
        

        boundary_mask = self.boundary_mask[side]
        j = 0
        for i in range(n):
            if mask[i]:
                full_boundary_array[i] = values[j]
                j += 1
            elif not boundary_mask[i]:
                full_boundary_array[i] = self.normal_wall_temp
            elif self.D[Room.walls_order[side]] is not None:
                full_boundary_array[i] = self.D[Room.walls_order[side]][i]
        return full_boundary_array


    def iterate_room(self):
        '''Update the room's temperature distribution.'''
        #Update boundary conditions from neighbors
        for coupling in self.neighbors:
            neighbor = coupling["neighbor"]
            side = coupling["side"]
            my_start = coupling["start"]
            my_end = coupling.get("end")
            their_start, their_end = neighbor.give_border_start_and_end(self)



            # Get the boundary values from the neighboring room
            neighbor_values = neighbor.get_boundary_value(
                Room.opposite_side(side), their_start, their_end
            )

            if neighbor_values is None:
                continue  # Skip if no values are returned

            #border_values = self.get_boundary_value(side, my_start, my_end)
            # Update the Neumann BC for this side based on the neighbor's values
            if coupling["type"] == "dirichlet":
                full_boundary_values = self.create_full_boundary_array(neighbor_values, side, my_start, my_end)
                self.D[Room.walls_order[side]] = full_boundary_values
            else:
                #neumann_boundary = (np.array(neighbor_values) - np.array(border_values)) / self.dx
                neumann_boundary = -np.array(neighbor_values)
                full_boundary_values = self.create_full_boundary_array(neumann_boundary, side, my_start, my_end)
                self.N[Room.walls_order[side]] = full_boundary_values

        self.solver.updateBC(self.D, self.N,self.u)

        self.new_u, _ = self.solver.solve()
        self.u = self.relaxation * self.new_u + (1 - self.relaxation) * self.u

if __name__ == "__main__":
    four = True  

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

    for _ in range(10):
        omega2.iterate_room()
        omega1.iterate_room()
        omega3.iterate_room()
        omega4.iterate_room() if four else None
        

    print("Room 1 Temperature Distribution:\n", omega1.u.reshape((omega1.Ny, omega1.Nx)))
    print("Room 2 Temperature Distribution:\n", omega2.u.reshape((omega2.Ny, omega2.Nx)))
    print("Room 3 Temperature Distribution:\n", omega3.u.reshape((omega3.Ny, omega3.Nx)))
    print("Room 4 Temperature Distribution:\n", omega4.u.reshape((omega4.Ny, omega4.Nx))) if four else None

    #plotting
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    im1 = axs[0].imshow(omega1.u.reshape((omega1.Ny, omega1.Nx)), cmap='hot', origin='lower', extent=[0, omega1.Lx, 0, omega1.Ly])
    axs[0].set_title('Room 1 Temperature Distribution')
    fig.colorbar(im1, ax=axs[0])
    im2 = axs[1].imshow(omega2.u.reshape((omega2.Ny, omega2.Nx)), cmap='hot', origin='lower', extent=[0, omega2.Lx, 0, omega2.Ly])
    axs[1].set_title('Room 2 Temperature Distribution')
    fig.colorbar(im2, ax=axs[1])
    im3 = axs[2].imshow(omega3.u.reshape((omega3.Ny, omega3.Nx)), cmap='hot', origin='lower', extent=[0, omega3.Lx, 0, omega3.Ly])
    axs[2].set_title('Room 3 Temperature Distribution')
    fig.colorbar(im3, ax=axs[2])
    im4 = axs[3].imshow(omega4.u.reshape((omega4.Ny, omega4.Nx)), cmap='hot', origin='lower', extent=[0, omega4.Lx, 0, omega4.Ly]) if four else None
    axs[3].set_title('Room 4 Temperature Distribution') if four else None
    fig.colorbar(im4, ax=axs[3]) if four else None
    plt.tight_layout()
    plt.show()