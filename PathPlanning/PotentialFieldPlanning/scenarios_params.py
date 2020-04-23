from utilities import load_image
import math


def define_scenario():
    # sx = 410.0  # start x position [m]
    # sy = 100.0  # start y positon [m]
    # gx = 120.0  # goal x position [m]
    # gy = 300.0  # goal y position [m]
    # grid, ox, oy = load_image('./curb_map2.png')

    sx = 190.0  # start x position [m]
    sy = 30.0  # start y positon [m]
    gx = 50.0  # goal x position [m]
    gy = 110.0  # goal y position [m]
    stheta = 0.5*math.pi
    grid, ox, oy = load_image('./curb1.png')
    # grid, ox, oy = load_image('./curb2.png')

    # sx = 190.0  # start x position [m]
    # sy = 70.0  # start y positon [m]
    # gx = 12.0  # goal x position [m]
    # gy = 50.0  # goal y position [m]
    # grid, ox, oy = load_image('./curb3.png')

    # sx = 20.0  # start x position [m]
    # sy = 0.0  # start y positon [m]
    # gx = 70.0  # goal x position [m]
    # gy = 30.0  # goal y position [m]
    # grid, ox, oy = load_image('./map_diag2.png')

    # sx = 40.0  # start x position [m]
    # sy = 2.0  # start y positon [m]
    # gx = 40.0  # goal x position [m]
    # gy = 70.0  # goal y position [m]
    # grid, ox, oy = load_image('./map.png')

    resolution = 1.0  # potential grid size [m]

    return sx, sy, stheta, gx, gy, grid, ox, oy, resolution
