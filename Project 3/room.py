import numpy as np

''' This is the way the axes for the coupling ("start") will be defined:
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
    '''
    A class to represent a room in the apartment.
    
    Attributes
    ----------
    name : str
        The name of the room (e.g., "omega1").
    shape : tuple
        The dimensions of the room (Lx, Ly).
    heater_sides : list
        The sides of the room with heaters (e.g., ["left", "right"]).
    window_sides : list
        The sides of the room with windows (e.g., ["top"]).'''




    def __init__(self, name, dx, shape, heater_sides, window_sides, heater_temp=40, window_temp=5, normal_wall_temp=15):
        self.name = name
        self.dx = dx
        self.Lx, self.Ly = shape
        self.heater_sides = heater_sides
        self.window_sides = window_sides
        self.heater_temp = heater_temp
        self.window_temp = window_temp
        self.normal_wall_temp = normal_wall_temp
        self.side_to_indices = {}
        self.neighbors = []  # List to store neighboring room couplings
        self.couplings = {}  # Dictionary to store coupling details by neighbor name

        self.initialize_BCs(*shape, heater_sides, window_sides)
        

    def initialize_BCs(self, Lx, Ly, heater_sides, window_sides):
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

        self.Nx = int(Lx / self.dx) + 1 #number of grid points in x direction
        self.Ny = int(Ly / self.dx) + 1 #number of grid points in y direction
        self.N = self.Nx * self.Ny

        #Boundary Indices
        self.side_to_indices["bottom"] = np.arange(0, self.Nx)
        self.side_to_indices["left"] = np.arange(0, self.N, self.Nx)
        self.side_to_indices["top"] = np.arange(self.N - self.Nx, self.N)
        self.side_to_indices["right"] = np.arange(self.Nx - 1, self.N, self.Nx)

        self.D_bottom = np.full(self.Nx, self.normal_wall_temp) #we set the normal wall as the default value
        self.D_left   = np.full(self.Ny, self.normal_wall_temp)
        self.D_top    = np.full(self.Nx, self.normal_wall_temp)
        self.D_right  = np.full(self.Ny, self.normal_wall_temp)

        #Apply window=5 and heater=40 overrides
        #window_sides is a list of strings that tells us which walls have windows
        #heater_sides is a list of strings that tells us which walls have heaters
        for s in window_sides:
            if   s == "bottom": D_bottom = np.full(self.Nx, self.window_temp)
            elif s == "left":   D_left   = np.full(self.Ny, self.window_temp)
            elif s == "top":    D_top    = np.full(self.Nx, self.window_temp)
            elif s == "right":  D_right  = np.full(self.Ny, self.window_temp)
        for s in heater_sides:
            if   s == "bottom": D_bottom = np.full(self.Nx, self.heater_temp)
            elif s == "left":   D_left   = np.full(self.Ny, self.heater_temp)
            elif s == "top":    D_top    = np.full(self.Nx, self.heater_temp)
            elif s == "right":  D_right  = np.full(self.Ny, self.heater_temp)

        #Do not use uniform_list yet! Otherwise you set the flux to zero
        self.N_bottom = None
        self.N_left   = None
        self.N_top    = None
        self.N_right  = None
    
    def add_coupling(self, coupling):
        '''Add a coupling to a neighboring room.
        Parameters
        ----------
            coupling : dict
                The neighboring room (e.g., {"neighbor" : <room_object>, "side": "bottom", "start": 0}).'''
        check_keys = {"neighbor", "side", "start"}
        if not isinstance(coupling, dict) or set(coupling.keys()) != check_keys:
            raise ValueError(f"Coupling must be a dictionary with keys: {', '.join(sorted(check_keys))}.")
        if not isinstance(coupling["neighbor"], room):
            raise ValueError("The 'neighbor' must be an instance of the room class.")
        if coupling["side"] not in {"bottom", "left", "top", "right"}:
            raise ValueError("The 'side' must be one of: bottom, left, top, right.")
        if not isinstance(coupling["start"], int) or coupling["start"] < 0:
            raise ValueError("The 'start' must be a non-negative integer.")
        self.neighbors.append(coupling)

    def get_boundary_value(self, side, start):
        '''Get the boundary value for a specific side and start index.
        Parameters
        ----------
            side : str
                The side of the room ("bottom", "left", "top", "right").
            start : int
                The starting index along the specified side.
        Returns
        -------
            numpy.ndarray or None
                The boundary value if set, otherwise None.'''
        if side == "bottom":
            if self.D_bottom is not None and 0 <= start < self.Nx:
                return self.D_bottom[start]
            elif self.N_bottom is not None and 0 <= start < self.Nx:
                return self.N_bottom[start]
        elif side == "left":
            if self.D_left is not None and 0 <= start < self.Ny:
                return self.D_left[start]
            elif self.N_left is not None and 0 <= start < self.Ny:
                return self.N_left[start]
        elif side == "top":
            if self.D_top is not None and 0 <= start < self.Nx:
                return self.D_top[start]
            elif self.N_top is not None and 0 <= start < self.Nx:
                return self.N_top[start]
        elif side == "right":
            if self.D_right is not None and 0 <= start < self.Ny:
                return self.D_right[start]
            elif self.N_right is not None and 0 <= start < self.Ny:
                return self.N_right[start]
        return None

if __name__ == "__main__":
    room1 = room("omega1", 0.05, (1,1), ["left","right"], ["top"])
    room2 = room("omega2", 0.05, (1,1), ["left"], ["top","right"])
    coupling1 = {"neighbor": room2, "side": "bottom", "start": 0}
    coupling2 = {"neighbor": room1, "side": "top", "start": 0}
    room1.add_coupling(coupling1)
    room2.add_coupling(coupling2)
    print(room1.neighbors)
    print(room2.neighbors)