from .visualization import *
from .dirichlet_neumann import DirichletNeumannSolverMPi
from .room import Room
from .add_boundaries import (
    create_inner_room,
    add_boundaries_room_1x1,
    add_boundaries_room_1x2,
    add_boundaries_room4_1x1
)
import numpy as np
import argparse



def main():
    """
    Main function to solve Project3.
    """
    # Parameters
    parser = argparse.ArgumentParser(description="Solve Project 3 simulation")
    parser.add_argument("--use-room4", action="store_true", help="Add the small room4 in the simulation")
    args = parser.parse_args()

    # ------ Change dx -------
    dx = 1 / 20
    # ------------------------

    room1 = Room(delta=dx, size_x=1, size_y=1)
    room2 = Room(delta=dx, size_x=1, size_y=2)
    room3 = Room(delta=dx, size_x=1, size_y=1)
    room4 = None
    if args.use_room4:
        room4 = Room(delta=dx, size_x=1 / 2, size_y=1 / 2)

    solver = DirichletNeumannSolverMPi(
        dx=dx, room1=room1, room2=room2, room3=room3, room4=room4, omega=0.8, n_iter=10
    )
    results = solver.iterate()
    T_normal = solver.T_normal
    T_h = solver.T_h
    T_wf = solver.T_wf

    grid_list = []

    if solver.rank == 0:
        if room4 is not None:
            u1_list, u2_list, u3_list, u4_list = results
        else:
            u1_list, u2_list, u3_list = results
        
        for i in range(len(u1_list)):
            T_room1 = create_inner_room(
                u1_list[i], Nx=room1.Nx_int + 1, Ny=room1.Ny_int
            )
            T_room1 = add_boundaries_room_1x1(
                T_room1, T_top=T_normal, T_bottom=T_normal, T_left=T_h
            )

            T_room2 = create_inner_room(u2_list[i], Nx=room2.Nx_int, Ny=room2.Ny_int)
            T_room2 = add_boundaries_room_1x2(
                T_room2, T_top=T_h, T_bottom=T_wf, T_right=T_normal, T_left=T_normal
            )

            T_room3 = create_inner_room(
                u3_list[i], Nx=room3.Nx_int + 1, Ny=room3.Ny_int
            )
            T_room3 = add_boundaries_room_1x1(
                T_room3, T_top=T_normal, T_right=T_h, T_bottom=T_normal
            )
            
            if room4 is not None:
                T_room4 = create_inner_room(u4_list[i], Nx=room4.Nx_int + 1, Ny=room4.Ny_int)
                T_room4 = add_boundaries_room4_1x1(T_room4, T_top=T_normal, T_right=T_normal, T_bottom=T_h)
                
                grid = merge_grids_by_position(
                    [
                        [0, room2.Nx - 1, T_room2],
                        [room2.Nx - 1, 0, T_room1],
                        [0, 2 * (room2.Nx - 1), T_room3],
                        [(room2.Nx - 1), 2*(room2.Nx - 1), T_room4]
                    ]
                )
            else:
                grid = merge_grids_by_position(
                    [
                        [0, room2.Nx - 1, T_room2],
                        [room2.Nx - 1, 0, T_room1],
                        [0, 2 * (room2.Nx - 1), T_room3],
                    ]
                )
            grid_list.append(grid)

        animate_grid2(grid_list, filename="animation_final.gif")


if __name__ == "__main__":
    main()
