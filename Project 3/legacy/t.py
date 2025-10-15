from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD.Clone()
rank = comm.Get_rank()
size = comm.Get_size()

'''
computation idea
3 rooms (Ω1, Ω2, Ω3)
All have heaters (Neumann BC)
Ω2 has a window (Dirichlet BC)
Ω1 and Ω3 connect to Ω2 (Dirichlet BC on Ω2, Neumann BC on Ω1 and Ω3)
Ω1 and Ω3 do not connect to each other
Ω1 and Ω3 are solved in parallel (rank 0 and rank 1)

Ω2 is solved in rank 0 after Ω1 and Ω3 are solved
'''

# Make sure that every send has a matching recv!
if rank == 0:
    # Send 10 numbers to rank 1 (dest = 1)
    # Method 'send' for Python objects (pickle under the hood)
    comm.send(data, dest=1)
    print(f"P[{rank}] sent data = {data}")

elif rank == 1:
    # Receive 10 numbers from rank 0 (source = 0)
    # Method 'recv' for Python objects (pickle under the hood)
    data = comm.recv(source=0)
    print(f"P[{rank}] received data = {data}")
