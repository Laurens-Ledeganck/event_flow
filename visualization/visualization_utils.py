"""
WORK IN PROGRESS
"""

# imports 
import pandas as pd
import numpy as np
import matplotlib.animation as anim
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from functools import partial
import imageio


def make_arrow(spread):
    return np.array([
        [-spread/20, 0, 0],
        [0, 0, 0],
        [-spread/20, spread/20, 0],
        [-spread/20, -spread/20, 0],
        [0, 0, 0]
    ])

def make_drone(spread):
    scale = spread / 50
    centers = [
        [1.5, 1.5],  # front left
        [-2, 2],  # back left
        [-2, -2],  # back right
        [1.5, -1.5]  # front right
    ]
    init_points = [
        [0.71, 0.71],  # front left
        [-1, 1],  # back left
        [-1, -1],  # back right
        [0.71, -0.71]  # front right
    ]
    points = []

    def draw_circle(i):
        nonlocal points
        points += [centers[i]]
        points += make_circle(centers[i][0], centers[i][1], init_points[i][0], init_points[i][1])

    # step 1: draw the front left circle
    points += [[0.6, 0.6]]
    draw_circle(0)
    points += [[1, 0.5]]

    # step 2: connect to and draw the back left circle
    points += [[-1, 1]]
    draw_circle(1)

    # step 3: connect to and draw the back right circle
    points += [[-1, -1]]
    draw_circle(2)

    # step 4: connect to and draw the front right circle
    points += [[0.6, -0.6]]
    draw_circle(3)
    points += [[0.6, -0.6], [1, -0.5],
               #[1, 0], [3, 0], [1, 0],  # add a 'pointer' at the front
               [1, 0.5]]

    points = np.hstack((np.array(points), np.zeros((len(points), 1))))
    points = points * scale
    return points

def make_circle(x_c, y_c, x_1, y_1, step=10):
    r = np.sqrt((x_c-x_1)**2 + (y_c-y_1)**2)
    init_angle = int(np.degrees(np.arctan((y_1-y_c)/(x_1-x_c))))
    if x_1 < x_c: init_angle += 180
    pts = [[x_1, y_1]]
    for angle in range(init_angle, init_angle+360, step):
        pts += [[x_c + r*np.cos(np.radians(angle)), y_c + r*np.sin(np.radians(angle))]]
    pts += [[x_1, y_1]]
    return pts

def print_progress(progress):
    bar_length = 20
    completed_blocks = int(progress * bar_length)
    remaining_blocks = bar_length - completed_blocks
    progress_bar = '[' + '#' * completed_blocks + '-' * remaining_blocks + ']'
    print('\r' + progress_bar, end='', flush=True)

