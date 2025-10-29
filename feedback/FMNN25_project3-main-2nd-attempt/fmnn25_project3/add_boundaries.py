import numpy as np


def create_inner_room(inner_values, Nx, Ny):
    """
    Builds the internal matrix of size (Ny, Nx) from a 1D vector.

    Example:
        inner_values = [u1, u2, u3, u4, u5, u6]
        Nx = 3, Ny = 2
        => [[u4, u5, u6],
            [u1, u2, u3]]

    Args:
        inner_values (list or np.ndarray): internal values in 1D form.
        Nx (int): number of columns.
        Ny (int): number of rows.

    Returns:
        np.ndarray: internal matrix (Ny x Nx).
    """
    inner_values = np.array(inner_values)

    # Check consistency between vector length and grid dimensions
    if len(inner_values) != Nx * Ny:
        raise ValueError(
            f"Number of values ({len(inner_values)}) does not match Nx*Ny = {Nx*Ny}."
        )

    # Split the vector into rows of length Nx
    rows = [inner_values[i * Nx : (i + 1) * Nx] for i in range(Ny)]

    # Reverse the vertical order so the last row becomes the top one
    inner_room = np.array(rows[::-1])

    return inner_room


def add_boundaries_room_1x1(
    inner_room, T_top=None, T_right=None, T_left=None, T_bottom=None
):
    """Add the boudaries for room1

    Args:
        inner_room1 : _description_
        T_top
    """
    room = np.copy(inner_room)
    Ny, Nx = room.shape

    # --- Add left boundary ---
    if T_left is not None:
        left_col = T_left * np.ones((Ny, 1))
        room = np.hstack([left_col, room])

    # --- Add right boundary ---
    if T_right is not None:
        right_col = T_right * np.ones((room.shape[0], 1))
        room = np.hstack([room, right_col])

    # --- Add top boundary ---
    if T_top is not None:
        top_row = T_top * np.ones((1, room.shape[1]))
        room = np.vstack([top_row, room])

    # --- Add bottom boundary ---
    if T_bottom is not None:
        bottom_row = T_bottom * np.ones((1, room.shape[1]))
        room = np.vstack([room, bottom_row])

    return room


def add_boundaries_room_1x2(
    inner_room, T_top=None, T_bottom=None, T_left=None, T_right=None
):
    """
    Add boundary conditions for room2 (1x2 configuration).

    The boundaries are automatically sized based on the inner_room shape.

    Args:
        inner_room (np.ndarray): internal temperature matrix (Ny x Nx)
        T_top (float or None): temperature at the top boundary
        T_bottom (float or None): temperature at the bottom boundary
        T_left (float or None): temperature at the left boundary
        T_right (float or None): temperature at the right boundary

    Returns:
        np.ndarray: temperature matrix including all boundaries
    """
    room = np.copy(inner_room)
    Ny, Nx = room.shape

    # --- Add top boundary ---
    if T_top is not None:
        top_row = T_top * np.ones((1, Nx))
        room = np.vstack([top_row, room])

    # --- Add bottom boundary ---
    if T_bottom is not None:
        bottom_row = T_bottom * np.ones((1, Nx))
        room = np.vstack([room, bottom_row])

    # --- Update shape after vertical stacking ---
    Ny, Nx = room.shape

    # --- Add left boundary ---
    if T_left is not None:
        left_col = T_left * np.ones((Ny, 1))
        room = np.hstack([left_col, room])

    # --- Add right boundary ---
    if T_right is not None:
        right_col = T_right * np.ones((Ny, 1))
        room = np.hstack([room, right_col])

    return room


def add_boundaries_room4_1x1(
    inner_room, T_top=None, T_right=None, T_left=None, T_bottom=None
):
    """Add the boudaries for room1

    Args:
        inner_room1 : _description_
        T_top
    """
    room = np.copy(inner_room)
    Ny, Nx = room.shape

    # --- Add top boundary ---
    if T_top is not None:
        top_row = T_top * np.ones((1, room.shape[1]))
        room = np.vstack([top_row, room])

    # --- Add bottom boundary ---
    if T_bottom is not None:
        bottom_row = T_bottom * np.ones((1, room.shape[1]))
        bottom_row[0][0] = T_top #The border is not a heater, so we correct it manually
        room = np.vstack([room, bottom_row])
        
    # --- Add left boundary ---
    if T_left is not None:
        left_col = T_left * np.ones((Ny, 1))
        room = np.hstack([left_col, room])

    # --- Add right boundary ---
    if T_right is not None:
        right_col = T_right * np.ones((room.shape[0], 1))
        room = np.hstack([room, right_col])


    return room