from .room import Room
from mpi4py import MPI
import numpy as np
from .matrix import matrix

matrix = matrix()

class DirichletNeumannSolverMPi:
    def __init__(self, omega=0.8, n_iter=10):
        self.omega = omega
        self.n_iter = n_iter
        
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        self.dx=1/3
        self.dx2= self.dx**2
        self.T_normal = 15
        self.T_h = 40
        self.T_wf = 5

        #Initialize rooms
        if self.rank == 0:
            self.room1 = Room(delta=1/3, size_x=1, size_y=1)
            #self.A1 = self.room1.A_neumann()
            self.A1 = matrix.A1
            self.T_room1_inside = 15*np.ones(((self.room1.Nx-1)*(self.room1.Ny-2)))
        
        if self.rank == 1:
            self.room2 = Room(delta=1/3, size_x=1, size_y=2)
            #self.A2 = self.room2.A_dirichlet()
            self.A2 = matrix.A2
            self.T_room2_inside = 15*np.ones((self.room2.Nx_int*self.room2.Ny_int))
                
        if self.rank == 2:
            self.room3 = Room(delta=1/3, size_x=1, size_y=1)
            #self.A3 = self.room3.A_neumann()
            self.A3 = matrix.A3
            self.T_room3_inside = 15*np.ones(((self.room3.Nx-1)*(self.room3.Ny-2)))
        
    
    def iterate(self):
        """Loop and compute Dirichlet-Neumann interaction with relaxation
        """
        if self.rank == 0:
            u1_list, u2_list, u3_list = [], [], []
            
        for it in range(self.n_iter):
            if self.rank == 1: # 2nd processor handling room 2
                #Receive interface values from neighbors
                gamma1 = self.comm.recv(source=0) # from room 1
                gamma2 = self.comm.recv(source=2) # from room 3
                
                # Build b2 vectors for room 2
                b2 = matrix.b2(T_bottom=self.T_wf, T_right_bottom=self.T_normal, gamma1=gamma1, T_top=self.T_h, T_left_top=self.T_normal, gamma2=gamma2, dx=self.dx)
                
                T_new = np.linalg.solve(self.A2, b2)
                
                #Relaxation
                self.T_room2_inside = self.omega*T_new + (1-self.omega)*self.T_room2_inside
                
                self.comm.send(self.T_room2_inside.copy(), dest=0)
                self.comm.send(self.T_room2_inside.copy(), dest=2)
                
                
                
            else:
                #Send interface values to room 2
                if self.rank == 0:
                    # Compute gamma1
                    indices_right = [(i+1)*self.room1.Nx_int - 1 for i in range(self.room1.Ny_int)]
                    interface_send = self.T_room1_inside[indices_right]
                else:
                    # Compute gamma2
                    indices_left = [i*self.room3.Nx_int for i in range(self.room3.Ny_int)]
                    interface_send = self.T_room3_inside[indices_left]

                #Send to room 2
                self.comm.send(interface_send, dest=1)
                
                #Receive updated T from room 2
                T_new = self.comm.recv(source=1)
                
                #build b1 and b3
                if self.rank == 0:
                    u2_list.append(T_new.copy())
                    q_right = np.array([ (T_new[0]-self.T_room1_inside[2]),
                             (T_new[2]-self.T_room1_inside[5])]) /self.dx
                    b1 = matrix.b1(dx=self.dx, T_bottom=self.T_normal, T_left=self.T_h, T_top=self.T_normal, q_right=q_right)
                    T_new = np.linalg.solve(self.A1, b1)
                    self.T_room1_inside = self.omega*T_new + (1-self.omega)*self.T_room1_inside
                    
                    u1_list.append(self.T_room1_inside.copy())
                    u3_list.append(self.comm.recv(source=2))
                    
                else:
                    q_left = np.array([ (T_new[7]-self.T_room3_inside[0]),
                             (T_new[9]-self.T_room3_inside[3])]) /self.dx
                    
                    b3 = matrix.b3(dx=self.dx, T_bottom=self.T_normal, T_top=self.T_normal, T_right=self.T_h, q_left=q_left)
                    T_new = np.linalg.solve(self.A3, b3)
                    self.T_room3_inside = self.omega*T_new + (1-self.omega)*self.T_room3_inside
                    self.comm.send(self.T_room3_inside.copy(), dest=0)
                    
        if self.rank == 0:
            return u1_list, u2_list, u3_list
        elif self.rank == 1:
            self.comm.send(self.T_room2_inside, dest=0)
        else:
            self.comm.send(self.T_room3_inside, dest=0)
    
        
if __name__ == "__main__":
    solver = DirichletNeumannSolverMPi(omega=0.8, n_iter=10)
    
    # Iterate
    results = solver.iterate()
    
    if solver.rank == 0:
        u1_list, u2_list, u3_list = results
        for i in range(len(u1_list)):
            print(f"\nIteration {i}")
            print("Room 1 Temperatures:", u1_list[i])
            print("Room 2 Temperatures:", u2_list[i])
            print("Room 3 Temperatures:", u3_list[i])