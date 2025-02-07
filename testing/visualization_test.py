import numpy as np
import matplotlib.pyplot as plt


def make_drone():
    scale = 1
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


pts = make_drone()
print(len(pts))
plt.plot(pts[:, 0], pts[:, 1])
plt.show()
