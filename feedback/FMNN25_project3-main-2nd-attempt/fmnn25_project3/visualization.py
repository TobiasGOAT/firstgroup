# -*- coding: utf-8 -*-
"""
Created on Sun Oct 12 23:48:49 2025

@author: Roland
"""
import unittest
import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter
import matplotlib.animation as animation


def open_file(filename):
    """Open a file with the default application, cross-platform."""
    if sys.platform.startswith("darwin"):  # macOS
        subprocess.call(("open", filename))
    elif os.name == "nt":  # Windows
        os.startfile(filename)
    elif os.name == "posix":  # Linux, Unix, etc.
        subprocess.call(("xdg-open", filename))


def merge_grids_3(room1=None, room2=None, room3=None, room4=None):
    """
    This function merges the grids of multiple rooms into one big grid.
    Empty grid cells are set to None, which is interpreted as white colour.
    The order of rooms is hardcoded.
    The merge works for 3 rooms or 4.

    Parameters
    ----------
    room1 : matrix
        The westmost room.
    room2 : matrix
        The eastmost room.
    room3 : matrix
        The middle room.
    room4 : matrix, optional
        Optional: the bottom half of the middle room.

    Returns
    -------
    grid : matrix of temperatures, with non-room cells set to None.

    """
    if room4 is None:
        room4 = np.zeros((0, 0))
    room1_width, room1_length = room1.shape
    room2_width, room2_length = room2.shape
    room3_width, room3_length = room3.shape
    room4_width, room4_length = room4.shape
    width = room1_width + room2_width + room3_width
    length = max(room1_length + room2_length, room3_length + 2)

    grid = np.full((width, length), None, dtype=float)
    grid[0:room1_width, 0:room1_length] = room1
    grid[room1_width : (room1_width + room4_width), 0:room4_length] = room4
    grid[
        room1_width : (room1_width + room3_width),
        length - room3_length - 1 : length - 1,
    ] = room3
    grid[(width - room2_width) : width, (length - room2_length) : length] = room2
    grid[room1_width : (room1_width + room3_width), 0] = np.full(room3_width, 5)
    grid[room1_width : (room1_width + room3_width), -1] = np.full(room3_width, 40)
    grid[room1_width - 1, room1_length:-1] = np.full(
        room3_length - room1_length - 2, 15
    )
    grid[width - room2_width, 1 : length - room2_length] = np.full(
        room3_length - room2_length - 2, 15
    )
    return grid


def merge_grids_by_position(grid_list):
    """
    This function merges the grids of multiple rooms into one big grid.
    Empty grid cells are set to None, which is interpreted as white colour.
    The order of rooms is hardcoded.
    The merge works for 3 rooms or 4.

    Parameters
    ----------
    grid_list: list of list(xpos, ypos, grid).
    Take xpos and ypos from the lower right corner.

    Returns
    -------
    grid : matrix of temperatures, with non-room cells set to None.

    """
    max_x = 0
    max_y = 0
    for i in range(len(grid_list)):
        x0, y0, grid = grid_list[i]
        top_x = x0 + grid.shape[0]
        top_y = y0 + grid.shape[1]
        max_x = max(max_x, top_x)
        max_y = max(max_y, top_y)

    new_grid = np.full((max_x, max_y), None, dtype=float)

    # print(f"x-size: {max_x}, y-size: {max_y}")
    for i in range(len(grid_list)):
        x0, y0, grid = grid_list[i]
        width, length = grid.shape
        # print(f"x: between {x0} and {x0+width}, width: {width},")
        # print(f"y: between {y0} and {y0+length}, length: {length}")
        new_grid[x0 : x0 + width, y0 : y0 + length] = grid
        # print(grid)
        # print(new_grid)

    return new_grid


def calculate_fontsize(grid):
    """
    Parameters
    ----------
    grid : matrix
        This calculates a proper fontsize.
        If the calculated fontsize is too small, it returns 0.

    Estimate font size based on grid size.

    Returns
    -------
    Integer value.

    """
    width, length = grid.shape
    fontsize = min(80 / width, 80 / length)
    if fontsize < 4:
        return 4
    if fontsize > 20:
        return 20
    return fontsize


def display_grid(grid):
    """
    Parameters
    ----------
    grid : matrix of temperatures
        This functions displays a grid of temperatures as a heatmap plot


    """
    width, length = grid.shape
    grid = np.rot90(grid, k=1)  # Turn the grid so that north is up.
    fig, ax = plt.subplots()
    im = ax.imshow(grid)
    fontsize = calculate_fontsize(grid)
    print(f"{width},{length}")
    # Loop over data dimensions and create text annotations.
    if width + length < 50:  # Skip text entirely if too small cells
        for x in range(width):
            for y in range(length):
                text = ax.text(
                    x,
                    y,
                    round(grid[y, x], 1),
                    ha="center",
                    va="center",
                    color="w",
                    fontsize=fontsize,
                )
                # print(f"{x},{y},{text}")

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("temperature", rotation=-90, va="bottom")
    ax.set_title("room heatmap")
    fig.tight_layout()
    plt.show()


def animate_grid(grid_list, filename="animation1.gif", fps=10, pad_ending=True):
    """
    Parameters
    ----------
    grid_list : list of matrices
        The input data.
    filename : filename for file, optional
        filename to save animation under. The default is "animation.gif".
    fps : Integer, optional
        Speed of animation, frames per second. The default is 10.
    pad_ending : boolean, optional
        Whether or not to add a pause at the end. The default is True.

    Returns
    -------
    None.

    """
    width, length = grid_list[0].shape
    fontsize = calculate_fontsize(grid_list[0])

    # Create initial figure & imshow
    fig, ax = plt.subplots(figsize=(6, 6))
    fig.tight_layout()

    # rotate grid to make north face up.
    grid = np.rot90(grid_list[0], k=1)
    im = ax.imshow(grid, vmin=5, vmax=40)

    # add a few extra frames in the end
    if pad_ending:
        for i in range(10):
            grid_list.append(grid_list[-1])

    texts = []
    if width + length < 50:
        for x in range(width):
            for y in range(length):
                t = ax.text(
                    x,
                    y,
                    f"{grid[y, x]:.1f}",  # fix rotation
                    ha="center",
                    va="center",
                    color="w",
                    fontsize=fontsize,
                )
                texts.append(t)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("temperature", rotation=-90, va="bottom")
    ax.set_title("room heatmap")

    # update colours and text for every grid in archive, save as frame
    writer = PillowWriter(fps=fps)
    with writer.saving(fig, filename, 100):
        for i in range(len(grid_list)):
            grid = np.rot90(grid_list[i], k=1)
            im.set_array(grid)
            k = -1
            for x in range(width):
                for y in range(length):
                    k += 1
                    texts[k].set_text(f"{grid[y, x]:.1f}")

            ax.set_title(f"room frame {i}")
            writer.grab_frame()

    for _ in range(20):  # Add extra frames to pause at the end of the gif
        writer.grab_frame()
    plt.close(fig)
    open_file(filename)


def animate_grid2(
    grid_list,
    filename="animation.gif",  # if None, do not save; otherwise save to this filename (e.g. "anim.gif")
    fps=10,
    pad_ending=False,
    rotate=0,
    interval=300,
    show_values=True,
    cmap_name="plasma",
):
    """
    Parameters
    ----------
    grid_list : list of 2D numpy arrays
        The input data (temperature grids).
    filename : str or None, optional
        Filename to save animation under (GIF). If None, animation is only shown.
    fps : int, optional
        Speed of animation, frames per second. The default is 10.
    pad_ending : bool, optional
        Whether or not to add a pause at the end. The default is True.
    rotate : int, optional
        Number of 90° rotations to apply to grids (0..3). The default is 0.
    interval : int, optional
        Time in milliseconds between frames for FuncAnimation. The default is 300.
    show_values : bool, optional
        Display numeric values inside cells when grid is small.
    cmap_name : str, optional
        Name of matplotlib colormap to use (default "plasma").

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
        The animation object (useful if you want to save or further manipulate).
    """

    # Prepare grids and orientation
    rotate = int(rotate) % 4
    grids = [np.rot90(np.array(g), k=rotate) for g in grid_list]

    # Optional padding at the end (pause)
    if pad_ending:
        for _ in range(10):
            grids.append(grids[-1])

    # determine global vmin/vmax ignoring NaNs so colors are stable between frames
    vmin = np.nanmin(np.array(grids))
    vmax = np.nanmax(np.array(grids))

    # colormap and NaN handling
    cmap = plt.cm.get_cmap(cmap_name).copy()
    cmap.set_bad(color="gray")

    # Keep same "background" as your working animate()
    fig, ax = plt.subplots()
    im = ax.imshow(grids[0], cmap=cmap, animated=True, vmin=vmin, vmax=vmax)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Temperature", rotation=-90, va="bottom")

    # prepare text labels if grid is small
    nrows, ncols = grids[0].shape
    texts = []
    if show_values and (nrows + ncols) < 50:
        # place texts at (col, row) with access grid[row, col]
        for r in range(nrows):
            for c in range(ncols):
                val = grids[0][r, c]
                if np.isnan(val):
                    txt = ax.text(
                        c,
                        r,
                        "",
                        ha="center",
                        va="center",
                        color="w",
                        fontsize=calculate_fontsize(grids[0]),
                    )
                else:
                    txt = ax.text(
                        c,
                        r,
                        f"{val:.1f}",
                        ha="center",
                        va="center",
                        color="w",
                        fontsize=calculate_fontsize(grids[0]),
                    )
                texts.append(txt)

    ax.set_title("Temperature Evolution")

    # update function for FuncAnimation
    def update(frame):
        grid = grids[frame]
        im.set_array(grid)
        if texts:
            k = 0
            for r in range(nrows):
                for c in range(ncols):
                    val = grid[r, c]
                    texts[k].set_text(f"{val:.1f}" if not np.isnan(val) else "")
                    k += 1
        ax.set_title(f"Temperature evolution - iteration {frame+1}")
        return [im] + texts if texts else [im]

    ani = animation.FuncAnimation(
        fig, update, frames=len(grids), interval=interval, blit=False, repeat=True
    )

    # Optionally save to GIF using PillowWriter (if filename provided)
    if filename is not None:
        try:
            writer = PillowWriter(fps=fps)
            ani.save(filename, writer=writer)
            print(f"Animation saved to {filename}")
        except Exception as e:
            print(f"Saving animation failed: {e}")

    plt.show()
    return ani


class TestVisualization(unittest.TestCase):
    """
    Since these tests are about making graphs,
    it makes less sense to have them in the test folder.
    """

    def test_merge(self):
        """test that grid merges work"""
        m1 = np.zeros((3, 3))
        m2 = np.full((3, 3), 5)
        m3 = np.full((3, 3), 4)
        m4 = np.full((3, 3), 2)
        merge1 = merge_grids_3(m1, m2, m3, m4)
        print("merge 1:\n", merge1)

        m1 = np.zeros((3, 3))
        m2 = np.full((3, 3), 5)
        m3 = np.full((3, 6), 2)
        merge2 = merge_grids_3(m1, m2, m3)
        print("merge 2:\n", merge2)

    def test_merge_advanced(self):
        """test that advanced grid merges work"""
        m1 = np.zeros((3, 3))
        m2 = np.full((3, 3), 6)
        m3 = np.full((3, 3), 5)
        m4 = np.full((3, 3), 3)
        grid_list = [[0, 0, m1], [3, 0, m4], [3, 3, m3], [6, 3, m2]]
        merge3 = merge_grids_by_position(grid_list)
        print("merge 3:\n", merge3)
        display_grid(merge3)

    def test_display(self):
        """test that the display function works"""
        m1 = np.zeros((3, 3))
        m2 = np.full((3, 3), 5)
        m3 = np.full((3, 3), 4)
        m4 = np.full((3, 3), 2)
        merge = merge_grids(m1, m2, m3, m4)
        # print("merge 1:\n", merge)
        display_grid(merge)

    def test_animate(self):
        """test that animation works"""
        m1 = np.zeros((3, 3))
        m2 = np.full((3, 3), 10)
        m3 = np.full((3, 3), 20)
        m4 = np.full((3, 3), 30)
        grid_list = [m1, m2, m3, m4]
        animate_grid(grid_list)


if __name__ == "__main__":
    unittest.main()
