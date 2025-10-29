from .room import Room
from mpi4py import MPI
import numpy as np


class DirichletNeumannSolverMPi:
    def __init__(
        self,
        room1: Room,
        room2: Room,
        room3: Room,
        dx,
        omega=0.8,
        n_iter=10,
        room4=None,
    ):
        self.omega = omega
        self.n_iter = n_iter

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        if room4 is not None:
            self.roomcount = 4
        else :
            self.roomcount = 3
            
        self.dx = dx
        self.dx2 = self.dx**2
        self.T_normal = 15
        self.T_h = 40
        self.T_wf = 5

        # Initialize rooms
        # Room 1
        self.room1 = room1
        self.A1 = self.room1.A_neumann(neumann_right=True)
        self.T_room1_inside = 15 * np.ones(((self.room1.Nx - 1) * (self.room1.Ny - 2)))

        # Room 2
        self.room2 = room2
        self.A2 = self.room2.A_dirichlet()
        self.T_room2_inside = 15 * np.ones((self.room2.Nx_int * self.room2.Ny_int))

        # Room 3
        self.room3 = room3
        self.A3 = self.room3.A_neumann(neumann_left=True)
        self.T_room3_inside = 15 * np.ones(((self.room3.Nx - 1) * (self.room3.Ny - 2)))

        if self.roomcount == 4:
            self.room4 = room4
            self.A4 = self.room4.A_neumann(neumann_left=True)
            self.T_room4_inside = 15 * np.ones(((self.room4.Nx - 1) * (self.room4.Ny - 2)))

    def iterate(self):
        """Loop and compute Dirichlet-Neumann interaction with relaxation"""
        if self.rank == 0:
            u1_list, u2_list, u3_list = [], [], []
            if self.roomcount==4:
                u4_list = []

        for it in range(self.n_iter):
            if self.rank == 1:  # 2nd processor handling room 2
                # Receive interface values from neighbors
                gamma1 = self.comm.recv(source=0)  # from room 1
                gamma2 = self.comm.recv(source=2)  # from room 3

                if self.roomcount ==4:
                    gamma3 = self.comm.recv(source=3) # from room 4
                    # Build b2 vectors for room 2
                    b2 = self.room2.b_dirichlet_two_by_one_4(T_bottom=self.T_wf, T_right_bottom=self.T_normal, gamma1=gamma1, T_top=self.T_h, T_left_top=self.T_normal, gamma2=gamma2, gamma3=gamma3)
                else:
                    # Build b2 vectors for room 2
                    b2 = self.room2.b_dirichlet_two_by_one(
                        gamma2=gamma2,
                        gamma1=gamma1,
                        T_bottom=self.T_wf,
                        T_top=self.T_h,
                        T_left_top=self.T_normal,
                        T_right_bottom=self.T_normal,
                    )
                T_new = np.linalg.solve(self.A2, b2)

                self.comm.send(T_new.copy(), dest=0, tag=2)
                self.comm.send(T_new.copy(), dest=2, tag=2)
                if self.roomcount==4:
                    self.comm.send(T_new.copy(), dest=3, tag=2)

                # Relaxation
                self.T_room2_inside = (
                    self.omega * T_new + (1 - self.omega) * self.T_room2_inside
                )

            else:
                # Send interface values to room 2
                if self.rank == 0:
                    if it == 0:
                        # Compute indices room 1
                        indices_gamma1_1 = [
                            (i + 1) * (self.room1.Nx_int + 1) - 1
                            for i in range(self.room1.Ny_int)
                        ]
                        indices_gamma1_2 = [
                            i * (self.room2.Nx_int) for i in range(self.room1.Ny_int)
                        ]
                    interface_send = self.T_room1_inside[indices_gamma1_1]
                elif self.rank == 2:
                    if it == 0:
                        # Compute indices room 3
                        indices_gamma2_3 = [
                            i * (self.room3.Nx_int + 1)
                            for i in range(self.room3.Ny_int)
                        ]
                        indices_gamma2_2 = [
                            (self.room2.Nx_int * self.room2.Ny_int - 1)
                            - i * (self.room3.Nx_int)
                            for i in range(self.room3.Ny_int)
                        ]
                        indices_gamma2_2 = indices_gamma2_2[::-1] #Reverse
                    interface_send = self.T_room3_inside[indices_gamma2_3]
                elif self.rank ==3:
                    if it == 0:
                        indices_gamma3_2 = [
                            ((self.room2.Nx_int * (self.room2.Ny_int//2)) - 1)
                            - i * (self.room3.Nx_int)
                            for i in range(self.room4.Ny_int)
                        ]
                        indices_gamma3_2 = indices_gamma3_2[::-1]
                        #indices_gamma3_2 = [i*(self.room2.Nx_int)-1 for i in [(self.room2.Ny_int//2)+k for k in range(self.room4.Ny_int)]]
                        indices_gamma3_4 = [i*(self.room4.Nx_int + 1) for i in range(self.room4.Ny_int)]
                    interface_send = self.T_room4_inside[indices_gamma3_4]

                self.comm.send(interface_send, dest=1)

                # Receive updated T from room 2
                T_new = self.comm.recv(source=1, tag=2)

                # build b1 and b3
                if self.rank == 0:
                    u2_list.append(T_new.copy())
                    q_right = (T_new[indices_gamma1_2] - interface_send) / self.dx
                    b1 = self.room1.b_neumann(
                        T_bottom=self.T_normal,
                        T_top=self.T_normal,
                        T_left=self.T_h,
                        q_right=q_right,
                    )
                    T_new = np.linalg.solve(self.A1, b1)
                    self.T_room1_inside = (
                        self.omega * T_new + (1 - self.omega) * self.T_room1_inside
                    )
                    u1_list.append(self.T_room1_inside.copy())
                    u3_list.append(self.comm.recv(source=2, tag=31))
                    if self.roomcount==4:
                        u4_list.append(self.comm.recv(source=3, tag=41))

                elif self.rank == 2:
                    q_left = (T_new[indices_gamma2_2] - interface_send) / self.dx
                    b3 = self.room3.b_neumann(
                        T_bottom=self.T_normal,
                        T_top=self.T_normal,
                        T_right=self.T_h,
                        q_left=q_left,
                    )
                    T_new = np.linalg.solve(self.A3, b3)
                    self.comm.send(T_new.copy(), dest=0, tag=31)
                    self.T_room3_inside = (
                        self.omega * T_new + (1 - self.omega) * self.T_room3_inside
                    )
                
                elif self.rank == 3:
                    q_left = (T_new[indices_gamma3_2]-interface_send)/self.dx
                    b4 = self.room4.b_neumann(T_bottom=self.T_h,T_top=self.T_normal, T_right=self.T_normal, q_left=q_left)
                    T_new = np.linalg.solve(self.A4, b4)
                    self.comm.send(T_new.copy(),dest=0, tag=41)
                    self.T_room4_inside = self.omega*T_new + (1-self.omega)*self.T_room4_inside

        if self.rank == 0:
            if self.roomcount == 4:
                return u1_list, u2_list, u3_list, u4_list
            else:
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
