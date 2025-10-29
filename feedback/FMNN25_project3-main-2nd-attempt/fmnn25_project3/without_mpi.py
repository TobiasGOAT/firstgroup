from .visualization import *
from .matrix import matrix


# Overwritte animate_grid for this specific test
def without_mpi():
    """
    Main function to solve Project3.
    """
    mat = matrix()
    A1 = mat.A1
    A2 = mat.A2
    A3 = mat.A3

    dx = 1 / 3
    dx2 = dx**2
    T_normal = 15
    T_h = 40
    T_wf = 5

    omega = 0.8

    T_room1_inside = 15 * np.ones(6)
    T_room2_inside = 15 * np.ones(10)
    T_room3_inside = 15 * np.ones(6)

    grid_list = []
    for i in range(10):

        if i > 0:
            T_room1_inside_previous = T_room1_inside.copy()
            T_room2_inside_previous = T_room2_inside.copy()
            T_room3_inside_previous = T_room3_inside.copy()

        print(f"Iteration {i}")
        print("----------")
        # ROOM 2
        gamma1 = np.array([T_room1_inside[2], T_room1_inside[5]])
        gamma2 = np.array([T_room3_inside[0], T_room3_inside[3]])
        b2 = mat.b2(
            T_bottom=T_wf,
            T_right_bottom=T_normal,
            gamma1=gamma1,
            T_top=T_h,
            T_left_top=T_normal,
            gamma2=gamma2,
            dx=dx,
        )
        T_room2_inside_new = np.linalg.solve(A2, b2)
        T_room2 = T_room2_inside_new.reshape(5, 2)[::-1, :]
        print("Room 2 Temperatures:")
        T_room2 = np.vstack([T_h * np.ones((1, 2)), T_room2, T_wf * np.ones((1, 2))])
        T_left = np.array(
            [
                [T_normal],
                [T_normal],
                [T_normal],
                [T_normal],
                [T_room1_inside[5]],
                [T_room1_inside[2]],
                [T_normal],
            ]
        )
        T_right = np.array(
            [
                [T_normal],
                [T_normal],
                [T_normal],
                [T_normal],
                [T_normal],
                [T_normal],
                [T_normal],
            ]
        )
        T_room2 = np.hstack([T_left, T_room2, T_right])
        print(T_room2)
        print(b2)
        # ROOM 1
        q_right = (
            np.array(
                [
                    (T_room2_inside_new[0] - T_room1_inside[2]),
                    (T_room2_inside_new[2] - T_room1_inside[5]),
                ]
            )
            / dx
        )
        b1 = mat.b1(dx, T_bottom=T_normal, T_left=T_h, T_top=T_normal, q_right=q_right)
        T_room1_inside_new = np.linalg.solve(A1, b1)
        T_room1 = np.array([T_room1_inside_new[3:6], T_room1_inside_new[0:3]])
        T_room1 = np.hstack([T_h * np.ones((2, 1)), T_room1])
        T_room1 = np.vstack(
            [T_normal * np.ones((1, 4)), T_room1, T_normal * np.ones((1, 4))]
        )

        print("Room 1 Temperatures:")
        print(T_room1)

        # ROOM 3
        q_left = (
            np.array(
                [
                    (T_room2_inside_new[7] - T_room3_inside[0]),
                    (T_room2_inside_new[9] - T_room3_inside[3]),
                ]
            )
            / dx
        )
        b3 = mat.b3(dx, T_bottom=T_normal, q_left=q_left, T_top=T_normal, T_right=T_h)
        T_room3_inside_new = np.linalg.solve(A3, b3)
        T_room3 = np.array([T_room3_inside_new[3:6], T_room3_inside_new[0:3]])
        T_room3 = np.hstack([T_room3, T_h * np.ones((2, 1))])
        T_room3 = np.vstack(
            [T_normal * np.ones((1, 4)), T_room3, T_normal * np.ones((1, 4))]
        )
        print("Room 3 Temperatures:")
        print(T_room3)

        if i > 0:
            T_room1_inside = (
                omega * T_room1_inside_new + (1 - omega) * T_room1_inside_previous
            )
            T_room2_inside = (
                omega * T_room2_inside_new + (1 - omega) * T_room2_inside_previous
            )
            T_room3_inside = (
                omega * T_room3_inside_new + (1 - omega) * T_room3_inside_previous
            )
        else:
            T_room1_inside = T_room1_inside_new
            T_room2_inside = T_room2_inside_new
            T_room3_inside = T_room3_inside_new

        grid = merge_grids_by_position(
            [[3, 0, T_room1], [0, 3, T_room2], [0, 6, T_room3]]
        )
        # display_grid(grid)
        grid_list.append(grid)

    animate_grid2(grid_list, filename="without_mpi.gif")


if __name__ == "__main__":
    without_mpi()
