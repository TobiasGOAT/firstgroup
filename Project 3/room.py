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
        The sides of the room with windows (e.g., ["top"]).
    couplings : dict
        The neighboring rooms (e.g., {"omega2": {"side": "bottom", "start": 0}}).'''




    def __init__(self, name, dx, shape, heater_sides, window_sides, couplings, heater_temp=40, window_temp=5, normal_wall_temp=15):
        self.name = name
        self.dx = dx
        self.Lx, self.Ly = shape
        self.heater_sides = heater_sides
        self.window_sides = window_sides
        self.couplings = couplings
        self.heater_temp = heater_temp
        self.window_temp = window_temp
        self.normal_wall_temp = normal_wall_temp
        self.dirichletBC, self.neumanBC = self.initialize_BCs(*shape, heater_sides, window_sides)

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
                        return
                    )

        self.Nx = int(Lx / self.dx) + 1 #number of grid points in x direction
        self.Ny = int(Ly / self.dx) + 1 #number of grid points in y direction
        self.N = self.Nx * self.Ny

        #Boundary Indices
        self.top = np.arange(0, self.Nx)
        self.left = np.arange(0, self.N, self.Nx)
        self.bottom = np.arange(self.N - self.Nx, self.N)
        self.right = np.arange(self.Nx - 1, self.N, self.Nx)

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

