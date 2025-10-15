import numpy as np
import heatSolver

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


class room:
    rooms_order = {"bottom": 0, "left": 1, "top": 2, "right": 3}
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

    def __init__(self, dx, shape, heater_sides=[], window_sides=[], heater_temp=40, window_temp=5, normal_wall_temp=15):
        self.dx = dx
        self.Lx, self.Ly = shape
        self.Nx = int(self.Lx / self.dx) + 1 #number of grid points in x direction
        self.Ny = int(self.Ly / self.dx) + 1 #number of grid points in y direction
        self.N = self.Nx * self.Ny
        self.heater_sides = heater_sides
        self.window_sides = window_sides
        self.heater_temp = heater_temp
        self.window_temp = window_temp
        self.normal_wall_temp = normal_wall_temp
        self.side_to_indices = {}
        self.neighbors = []  # List to store neighboring room couplings 
        self.couplings = {}  # Dictionary to store coupling details by neighbor name
        self.u = np.zeros(self.N)  # Initialize temperature array
        self.D = [None, None, None, None]  # Dirichlet BCs: [bottom, left, top, right]
        self.N = [None, None, None, None]  # Neumann BCs: [bottom, left, top, right]

        self.initialize_BCs(*shape, heater_sides, window_sides)
        
        self.solver = heatSolver(self.dx, (self.Lx, self.Ly), self.D, self.N)


    def _opposite_side(self, side):
        opposites = {
            "bottom": "top",
            "top": "bottom",
            "left": "right",
            "right": "left"
        }
        return opposites.get(side, None)
    
    def initialize_BCs(self, heater_sides, window_sides):
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
        self.side_to_indices["left"] = np.arange(0, self.N, self.Nx)
        self.side_to_indices["top"] = np.arange(self.N - self.Nx, self.N)
        self.side_to_indices["right"] = np.arange(self.Nx - 1, self.N, self.Nx)

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

        #Do not use uniform_list yet! Otherwise you set the flux to zero
        self.N[0] = None
        self.N[1] = None
        self.N[2] = None
        self.N[3] = None

    def add_coupling(self, coupling):
        """
        Adds a coupling between this room and a neighboring room.

        Parameters
        coupling : dict
            A dictionary specifying the coupling details. Must contain the following keys:
            - "neighbor": An instance of the neighboring room (must be of type `room`).
            - "side": A string indicating which side of the current room is coupled ("bottom", "left", "top", "right").
            - "start": Length of the start along the specified side from NEIGHBOR's perspective.
            - "end": Length of the end along the specified side from NEIGHBOR's perspective.


        Raises
        ------
        ValueError
            If the coupling dictionary is missing required keys, contains invalid values, or types are incorrect.

        Example
        -------
        >>> coupling = {
        ...     "neighbor": room_instance,
        ...     "side": "left",
        ...     "start": 0,
        ...     "end": 5
        ... }
        >>> room.add_coupling(coupling)
        """
        check_keys = {"neighbor", "side", "start", "end"}
        if not isinstance(coupling, dict) or set(coupling.keys()) != check_keys:
            raise ValueError(f"Coupling must be a dictionary with keys: {', '.join(sorted(check_keys))}.")
        if not isinstance(coupling["neighbor"], room):
            raise ValueError("The 'neighbor' must be an instance of the room class.")
        if coupling["side"] not in {"bottom", "left", "top", "right"}:
            raise ValueError("The 'side' must be one of: bottom, left, top, right.")
        if not isinstance(coupling["start"], int) or coupling["start"] < 0:
            raise ValueError("The 'start' must be a non-negative integer.")
        if "end" in coupling and (not isinstance(coupling["end"], int) or coupling["end"] <= coupling["start"]):
            raise ValueError("The 'end' must be an integer greater than 'start'.")
        self.neighbors.append(coupling)

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

        return [self.u[i] for i in self.side_to_indices[side][start:end]]
    
    def iterate_room(self):
        '''Update the room's temperature distribution.'''
        #Update boundary conditions from neighbors
        for coupling in self.neighbors:
            neighbor = coupling["neighbor"]
            side = coupling["side"]
            start = coupling["start"]
            end = coupling.get("end")

            # Get the boundary values from the neighboring room
            neighbor_values = neighbor.get_boundary_value(
                self._opposite_side(side), start, end
            )

            if neighbor_values is None:
                continue  # Skip if no values are returned

            # get the values for the neighboring room
            border_values = [self.u[i] for i in self.side_to_indices[side][start:end]]
            # Update the Neumann BC for this side based on the neighbor's values
            self.N[self.rooms_order[side]] = (np.array(border_values) - np.array(neighbor_values)) / self.dx
        
        self.solver.update_BCs(None, self.N)

        self.u = self.solver.solve(True)

if __name__ == "__main__":
    omega1 = room(0.1, (1.0, 1.0), heater_sides=["left"])
    omega2 = room(0.1, (1.0, 2.0), heater_sides=["top"], window_sides=["bottom"])
    omega3 = room(0.1, (1.0, 1.0), window_sides=["right"])

    omega1.add_coupling({"neighbor": omega2, "side": "right", "start": 0, "end": 1.0})
    omega2.add_coupling({"neighbor": omega1, "side": "left", "start": 0, "end": 1.0})
    omega2.add_coupling({"neighbor": omega3, "side": "right", "start": 1.0, "end": 2.0})
    omega3.add_coupling({"neighbor": omega2, "side": "left", "start": 0, "end": 1.0})