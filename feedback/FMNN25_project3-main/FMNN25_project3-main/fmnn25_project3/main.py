from .visualization import *
from .dirichlet_neumann import DirichletNeumannSolverMPi
import numpy as np


def main():
    """
    Main function to solve Project3.
    """
    solver = DirichletNeumannSolverMPi(omega=0.8, n_iter=10)
    
    # Iterate
    results = solver.iterate()
    T_normal = solver.T_normal
    T_h = solver.T_h
    T_wf = solver.T_wf
    
    grid_list = []
    
    if solver.rank == 0:
        u1_list, u2_list, u3_list = results
        T1_list, T2_list, T3_list = [], [], []
        for i in range(len(u1_list)):
            T_room1 = np.array([u1_list[i][3:6], u1_list[i][0:3]])
            T_room1 = np.hstack([solver.T_h * np.ones((2, 1)), T_room1])
            T_room1 = np.vstack([solver.T_normal * np.ones((1, 4)), T_room1, solver.T_normal * np.ones((1, 4))])
            T1_list.append(T_room1)

            T_room2 = u2_list[i].reshape(5, 2)[::-1, :]
            T_room2 = np.vstack([T_h * np.ones((1, 2)), T_room2, T_wf * np.ones((1, 2))])
            T_left = np.array([[T_normal],[T_normal],[T_normal],[T_normal],[u1_list[i][5]],[u1_list[i][2]],[T_normal],])
            T_right = np.array([[T_normal],[T_normal],[T_normal],[T_normal],[T_normal],[T_normal],[T_normal],])
            T_room2 = np.hstack([T_left, T_room2, T_right])
            T2_list.append(T_room2)
            
            T_room3 = np.array([u3_list[i][3:6], u3_list[i][0:3]])
            T_room3 = np.hstack([T_room3, T_h * np.ones((2, 1))])
            T_room3 = np.vstack([T_normal * np.ones((1, 4)), T_room3, T_normal * np.ones((1, 4))])
            T3_list.append(T_room3)
            
            grid = merge_grids_by_position(
            [[3, 0, T_room1], [0, 3, T_room2], [0, 6, T_room3]]
            )
            grid_list.append(grid)
            
        animate_grid2(grid_list, filename="animation_final.gif")
        
if __name__ == "__main__":
    main()
