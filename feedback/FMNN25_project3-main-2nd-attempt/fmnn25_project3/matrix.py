import numpy as np


class matrix:
    def __init__(self):
        # Room1
        self.A1 = (
            np.array(
                [
                    [-4, 1, 0, 1, 0, 0],
                    [1, -4, 1, 0, 1, 0],
                    [0, 1, -3, 0, 0, 1],
                    [1, 0, 0, -4, 1, 0],
                    [0, 1, 0, 1, -4, 1],
                    [0, 0, 1, 0, 1, -3],
                ]
            )
            * 9
        )

        # Room2
        self.A2 = (
            np.array(
                [
                    [-4, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, -4, 0, 1, 0, 0, 0, 0, 0, 0],
                    [1, 0, -4, 1, 1, 0, 0, 0, 0, 0],
                    [0, 1, 1, -4, 0, 1, 0, 0, 0, 0],
                    [0, 0, 1, 0, -4, 1, 1, 0, 0, 0],
                    [0, 0, 0, 1, 1, -4, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0, -4, 1, 1, 0],
                    [0, 0, 0, 0, 0, 1, 1, -4, 0, 1],
                    [0, 0, 0, 0, 0, 0, 1, 0, -4, 1],
                    [0, 0, 0, 0, 0, 0, 1, 1, 1, -4],
                ]
            )
            * 9
        )

        # Room3
        self.A3 = (
            np.array(
                [
                    [-3, 1, 0, 1, 0, 0],
                    [1, -4, 1, 0, 1, 0],
                    [0, 1, -4, 0, 0, 1],
                    [1, 0, 0, -3, 1, 0],
                    [0, 1, 0, 1, -4, 1],
                    [0, 0, 1, 0, 1, -4],
                ]
            )
            * 9
        )

    def b1(self, dx, T_bottom, T_left, T_top, q_right):
        dx2 = dx**2
        b = np.array(
            [
                (-T_bottom - T_left) / dx2,
                (-T_bottom) / dx2,
                -T_bottom / dx2 - q_right[0] / dx,
                (-T_left - T_top) / dx2,
                (-T_top) / dx2,
                -T_top / dx2 - q_right[1] / dx,
            ]
        )
        return b

    def b2(self, T_bottom, T_right_bottom, gamma1, T_top, T_left_top, gamma2, dx):
        dx2 = dx**2
        b = np.array(
            [
                (-gamma1[0] - T_bottom),
                (-T_bottom - T_right_bottom),
                (-gamma1[1]),
                (-T_right_bottom),
                (-T_left_top),
                (-T_right_bottom),
                (-T_left_top),
                (-gamma2[0]),
                (-T_top - T_left_top),
                (-gamma2[1] - T_top),
            ]
        )

        return b / dx2

    def b3(self, dx, T_bottom, q_left, T_top, T_right):
        dx2 = dx**2
        b = np.array(
            [
                (-T_bottom) / dx2 - q_left[0] / dx,
                -T_bottom / dx2,
                (-T_bottom - T_right) / dx2,
                -q_left[1] / dx - T_top / dx2,
                -T_top / dx2,
                (-T_top - T_right) / dx2,
            ]
        )
        return b
