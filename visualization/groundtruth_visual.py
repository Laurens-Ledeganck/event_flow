"""
This code makes a visualization of the ground truth.
For detailed requirements, see the settings at the top of the file. 
"""

# settings
data_dir = '../testing/indoor_forward_9'  # path to the directory where the relevant files are located
ground_truth_file = 'groundtruth.txt'  # file with the ground truth values
ground_truth_cols = ['t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']  # column names for the ground truth file;
# these have to include 't', 'x', 'y', 'z', and also 'qx', 'qy' and 'qz'

rotation = True  # specify whether to include an arrow indicating the rotation of the drone
noise_margin = 1.  # m, tune this to find the right orientation

images = True  # specify whether to include images;
# setting this to False makes the next variables redundant
image_file = 'images.txt'  # file with the images 
image_cols = ['id', 't', 'filename']  # column names for the image file; 
# these have to include 't' and 'filename'

events = True  # specify whether to include event frames;
# setting this to False makes the next variables redundant
event_file = 'events.txt'
event_cols = ['t', 'x', 'y', 'p']  # column names for the event file;
# these have to include 't', 'x' and 'y'. If 'p' is missing, a polarity of 1 is assumed for all events.
event_window = 0.001  # the 'shutter time' for an event image; if None this will be set to dt;
# 0.001-0.01 seems to be a good range

# TODO: fix the IMU stuff
imu = False  # specify whether to include IMU data (only works if images or events are also enabled);
# setting this to False makes the next variables redundant
imu_file = 'imu.txt'
imu_cols = ['t', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'lin_acc_x', 'lin_acc_y', 'lin_acc_z']

focus = True  # specify whether to center the current location in the view
zoom = 50  # %, set to 100 for focused view, set to 200 for detail view, set to 50 for zoomed out view
save = True  # save as well as plot
file_to_write = 'indoor_forward_9.mp4'  # name of the output file
dt = 0.01  # s, the time between each frame
speed = 1  # the speed for the saved video wrt the original speed


# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import imageio

from visualization_utils import *


# initialization
def initialize():
    orienter, init_rot, imgs, im_ax, evts, ev_ax, imu_axs, accs = None, None, None, None, None, None, None, None
    
    # load the ground truth data
    data = np.genfromtxt(data_dir + '/' + ground_truth_file, delimiter=' ', skip_header=1)
    start_time, end_time = float(data[0, ground_truth_cols.index('t')]), float(data[-1, ground_truth_cols.index('t')])

    # set the initial time to 0
    data[:, ground_truth_cols.index('t')] = np.round( data[:, ground_truth_cols.index('t')] - start_time , 3)
    
    # set the initial position to 0
    x, y, z = data[0][ground_truth_cols.index('x')], data[0][ground_truth_cols.index('y')], data[0][ground_truth_cols.index('z')]
    arr = np.zeros(data.shape[1])
    arr[ground_truth_cols.index('x')] = x
    arr[ground_truth_cols.index('y')] = y
    arr[ground_truth_cols.index('z')] = z
    data = data - arr

    fig, ax, im_ax, ev_ax, imu_axs, limits, spread = initialize_plots(data, images)

    if rotation:
        orienter, init_rot = initialize_rotation(data, spread)

    if events:
        initialize_events(data)

    if images:
        imgs = initialize_images(start_time)

    if imu:
        accs = initialize_imu(start_time)

    print('Using data from ', data_dir,
          '(', round(end_time - start_time, 3), 's)')
    return data, fig, ax, limits, ev_ax, im_ax, imgs, imu_axs, accs, orienter, init_rot, start_time, spread


def initialize_plots(data, images):
    fig = plt.figure()
    im_ax, ev_ax, imu_axs = None, None, None

    if events or images:
        if imu:
            imu_axs = [fig.add_subplot(461), fig.add_subplot(462), fig.add_subplot(463),
                       fig.add_subplot(467), fig.add_subplot(468), fig.add_subplot(469)]
            ax = fig.add_subplot(223, projection='3d')
        else:
            ax = fig.add_subplot(121, projection='3d')

        if events and images:
            ev_ax = fig.add_subplot(222)
            im_ax = fig.add_subplot(224)

        elif images:
            im_ax = fig.add_subplot(122)

        else:
            ev_ax = fig.add_subplot(122)
    else:
        ax = fig.add_subplot(projection='3d')

    xyz = np.vstack((data[:, ground_truth_cols.index('x')], data[:, ground_truth_cols.index('y')], data[:, ground_truth_cols.index('z')]))
    limits = np.vstack((np.min(xyz, axis=1), np.max(xyz, axis=1), (np.max(xyz, axis=1)+np.min(xyz, axis=1))/2, (np.max(xyz, axis=1)-np.min(xyz, axis=1))))
    spread = 1.1 * max(limits[3, :])

    round_to = 2  # significant figures
    round_to = round_to - int(np.floor(np.log10(abs(spread)))) - 1

    limits = np.round(np.vstack((limits[2, :] - spread/2, limits[2, :] + spread/2)), round_to).T
    limits = np.vstack((limits, np.array([[0, 360], [0, 270]])))  # TODO tune automatically

    return fig, ax, im_ax, ev_ax, imu_axs, limits, spread


def initialize_rotation(data, spread):
    #orienter = make_arrow(spread)
    orienter = make_drone(spread)

    # set initial rotation to zero
    qx, qy, qz, qw = data[0][ground_truth_cols.index('qx')], data[0][ground_truth_cols.index('qy')], data[0][ground_truth_cols.index('qz')], data[0][ground_truth_cols.index('qw')]
    rot = R.from_quat([qx, qy, qz, qw])  # initial offset
    orienter = rot.inv().apply(orienter)  # compensation
    # the above pre-rotates the arrow to compensate for the initial rotation offset

    # determine initial orientation
    init_rot = find_orientation(data, noise_margin)

    return orienter, init_rot


def find_orientation(data, noise_margin):
    # small loop to find the correct i (should be faster than numpy search functions)
    for i in range(len(data)):
        if (abs(data[i][ground_truth_cols.index('x')] - data[0][ground_truth_cols.index('x')]) >= noise_margin
           or abs(data[i][ground_truth_cols.index('y')] - data[0][ground_truth_cols.index('y')]) >= noise_margin):
            break

    # find the initial direction in x and y
    init_dir = np.array([data[i][ground_truth_cols.index('x')] - data[0][ground_truth_cols.index('x')], data[i][ground_truth_cols.index('y')] - data[0][ground_truth_cols.index('y')], 0])
    x, y = init_dir[0] / np.linalg.norm(init_dir), init_dir[1] / np.linalg.norm(init_dir)

    # return corresponding rotation matrix assuming rotation about z-axis
    return R.from_matrix(np.array([
        [x, -y, 0],
        [y, x, 0],
        [0, 0, 1]
    ]))


def initialize_events(evts):
    global event_cols, event_window, dt, ev_idx

    if 'p' not in event_cols:
        evts = np.hstack((evts, np.ones((len(evts), 1))))
        event_cols += ['p']

    if not event_window:
        event_window = dt
    print('event_window = ', event_window)

    ev_idx = 0


def initialize_images(start_time):
    imgs = pd.read_csv(data_dir + '/' + image_file, delimiter=' ', skiprows=1, names=image_cols, index_col=False)
    imgs.loc[:, 't'] = np.round(imgs.loc[:, 't'] - start_time, 3)
    return imgs


def initialize_imu(start_time):
    accs = np.genfromtxt(data_dir + '/' + imu_file, delimiter=' ', skip_header=1)
    accs[:, imu_cols.index('t')] = np.round(accs[:, imu_cols.index('t')] - start_time, 3)
    return accs


# updates
def update(i, t, data, fig, ax, limits, ev_ax, im_ax, imgs, imu_axs, accs, orienter, init_rot, spread):
    ax.clear()

    # update position
    x, y, z = data[i][ground_truth_cols.index('x')], data[i][ground_truth_cols.index('y')], data[i][ground_truth_cols.index('z')]
    ax.scatter(x, y, z, label='current location', c='b', marker='o')
    # plot the trajectory
    ax.plot(data[:i+1, ground_truth_cols.index('x')], data[:i+1, ground_truth_cols.index('y')], data[:i+1, ground_truth_cols.index('z')], label='trajectory', c='b')

    if focus:
        limits = np.array([
            [x - 1.1 * spread / (zoom/10), x + 1.1 * spread / (zoom/10)],
            [y - 1.1 * spread / (zoom/10), y + 1.1 * spread / (zoom/10)],
            [z - 1.1 * spread / (zoom/10), z + 1.1 * spread / (zoom/10)],
        ])
    ax.set_xlim(limits[0][0], limits[0][1])
    ax.set_ylim(limits[1][0], limits[1][1])
    ax.set_zlim(limits[2][0], limits[2][1])

    if rotation:
        update_rotation(data, orienter, init_rot, ax, x, y, z)

    if events:
        ev_ax = update_events(t, ev_ax)

    if images:
        im_ax = update_image(t, dt, im_ax, imgs)

    if imu:
        imu_axs = update_imu(i, imu_axs, accs)

    fig.tight_layout()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    img = img[:, :, :3]
    return img


def update_rotation(data, arrow, init_rot, ax, x, y, z):
    qx, qy, qz, qw = data[i][ground_truth_cols.index('qx')], data[i][ground_truth_cols.index('qy')], data[i][
        ground_truth_cols.index('qz')], data[i][ground_truth_cols.index('qw')]
    rot = R.from_quat([qx, qy, qz, qw])
    rotated_obj = rot.apply(arrow)
    rotated_obj = init_rot.apply(rotated_obj)
    translated_obj = rotated_obj + np.array([x, y, z])

    ax.plot(translated_obj[:, 0], translated_obj[:, 1], translated_obj[:, 2], c='black')

    return ax


def update_events(t, ev_ax):
    global ev_idx, start_time
    ev_ax.clear()

    e = []
    with open(data_dir + '/' + event_file, 'r') as file:
        if ev_idx:
            file.seek(ev_idx)
        else:
            ev_idx = 209598478
            file.seek(ev_idx)
        line = file.readline()
        ev_stop = False
        while not ev_stop:
            values = line.split()
            if float(values[event_cols.index('t')]) >= t + start_time:
                values[event_cols.index('y')] = 270 - float(values[event_cols.index('y')])  # TODO tune automatically
                e += [values]
            line = file.readline()
            if float(values[event_cols.index('t')]) >= t + event_window + start_time or line is None:
                ev_stop = True
                ev_idx = file.tell()
    e = np.array(e, dtype=float)

    colors = ('blue', 'red')  # colors for positive and negative events respectively
    clrs = [colors[int(e[i, event_cols.index('p')])] for i in range(e.shape[0])]
    ev_ax.set_facecolor('black')
    ev_ax.scatter(e[:, event_cols.index('x')], e[:, event_cols.index('y')], color=clrs, s=1)
    ev_ax.set_xlim(limits[3, 0], limits[3, 1])
    ev_ax.set_ylim(limits[4, 0], limits[4, 1])

    return ev_ax


def update_image(t, dt, im_ax, imgs):
    idx = np.where(np.round(imgs['t'].to_numpy() / dt) == round(t / dt))[0]
    img_path = imgs.loc[idx, 'filename']
    if len(img_path) > 0:
        img_path = img_path.values[0]
        _img = plt.imread(data_dir + '/' + img_path)
        im_ax.clear()
        im_ax.imshow(_img)

    return im_ax


def update_imu(i, imu_axs, accs):
    for i, axes in enumerate(imu_axs):
        axes.plot(accs[:i, 0], accs[:i, i+1])  # TODO tune automatically
        axes.set_title(imu_cols[i])
    return imu_axs


# main loop
data, fig, ax, limits, ev_ax, im_ax, imgs, imu_axs, accs, orienter, init_rot, start_time, spread = initialize()

if save:
    frame_rate = int(speed * 1/dt)  # frames per second
    frames = []
    t = 0
    end = data[-1, ground_truth_cols.index('t')]
    while t <= end:
        i = np.where(np.round(data[:, ground_truth_cols.index('t')] / dt) == round(t / dt))[0][0]
        frames += [update(i, t, data, fig, ax, limits, ev_ax, im_ax, imgs, imu_axs, accs, orienter, init_rot, spread)]

        print_progress(t/end)
        plt.pause(dt)
        t += dt

    frames = frames[1:]
    imageio.mimsave(file_to_write, frames, 'mp4', fps=frame_rate)

else:
    t = 0
    end = data[-1, ground_truth_cols.index('t')]
    while t <= end:
        try:
            i = np.where(np.round(data[:, 0]/dt) == round(t/dt))[0][0]
            update(i, t, data, fig, ax, limits, ev_ax, im_ax, imgs, imu_axs, accs, orienter, init_rot, spread)

        except Exception as e:
            print('Could not find entry at time ', t, '\n', e)

        plt.pause(dt)
        t += dt

print('Done.', flush=True)
