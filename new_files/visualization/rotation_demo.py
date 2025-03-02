"""
WORK IN PROGRESS

Tomorrow: 
1 - reshape matrix outputs
2 - train proper model using euler_deg and init & one using quat and absolute

Main issues:
1) Fixing code
!    - reshape matrix outputs
2) Error discrepancy
    -> issue mostly resolved, errors ~e-6 in testing vs ~e-7 in training, still different values for the first one though
    - difference in first iteration
3) Better model 
    - quat targets
    - matrix targets
    - euler angles on body axes
!    - the drone always turns left! -> data augmentation
    - maybe redefine loss so it's at order 10^0
    - try training on rotation matrix -> flawless_gnat_521, lr is too high, might not be reshaping matrix -> trusting_toad_636
    - try training on euler_deg -> suave_conch_448
    - RQ: What is the best form for the targets? 
    -> rotvec values ~e-4, matrix values ~e-1, euler values (deg) ~e-2, quat values ~e-1
    -> intuitively, I'd expect euler angles to be easiest -> reasonable magnitudes, easy to relate to optic flow

TODO: 
 - clean up code
 - make option to just use body axes
( - make drone visual better)
 - more rotation files
 - figure out how to ensure files are read in the right order
 - add dedicated config file
 - create a proper eval_rotation file
"""

"""
This code makes a visualization of the ground truth. And the model prediction. 
For detailed requirements, see the settings at the top of the file. 

Note: for the paths, assume event_flow directory
"""

# settings
config_file = 'configs/train_flow.yml'
model_file = 'results/mlruns/2025-feb/model-name/artifacts/model/data/model.pth'
model_file = model_file.replace('model-name', 'respected-koi-704')  # eg 'suave-conch-448'  or  'trusting-toad-636' 
data_dir = 'datasets/data/rotation_demo'

#data_dir = 'testing/indoor_forward_3'  # path to the directory where the relevant files are located
# event_file = 'events.txt'  # file with the event data
# event_cols = ['t', 'x', 'y', 'p']  # column names for the event file;
# these have to include 't', 'x' and 'y'. If 'p' is missing, a polarity of 1 is assumed for all events.
# event_window = 0.01  # s, the 'shutter time' for an event image; if None this will be set to dt;
# 0.001-0.01 seems to be a good range
# ground_truth_file = 'groundtruth.txt'  # file with the ground truth values
# ground_truth_cols = ['t', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw']  # column names for the ground truth file;
# these have to include 't', 'x', 'y', 'z', and also 'qx', 'qy' and 'qz'
noise_margin = 1.  # m, tune this to find the right orientation

save = True  # save as well as plot, setting to False will enable logging
file_to_write = 'model_indoor_forward_3_test.mp4'  # name of the output file
dt = 0.01  # s, the time between each frame
speed = 1  # the speed for the saved video wrt the original speed

# do not modify the following: 
rotation = True 
focus = True  
zoom = 200  
images = False 
events = False 

# TODO: include IMU stuff
imu = False  # specify whether to include IMU data (only works if images or events are also enabled);
# setting this to False makes the next variables redundant
# imu_file = 'imu.txt'
# imu_cols = ['t', 'ang_vel_x', 'ang_vel_y', 'ang_vel_z', 'lin_acc_x', 'lin_acc_y', 'lin_acc_z']


# imports
import torch
import torchvision.transforms.v2 as transforms
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import imageio
import h5py

import os
import sys
project_dir_name = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(project_dir_name)
from visualization_utils import *
from new_files.rotation_utils import ModifiedH5Loader
from configs.parser import YAMLParser


# initialization
def initialize():  # adapted from the other file
    orienter, init_rot, imgs, im_ax, evts, ev_ax, imu_axs, accs = None, None, None, None, None, None, None, None
    
    # load the ground truth data
    config_parser = YAMLParser(config_file)
    config = config_parser.config
    config = config_parser.combine_entries(config)

    # reconfigure settings, TODO: do this more cleanly
    config["data"]["path"] = data_dir  # use a limited amount of data for the demo
    config["loader"]["batch_size"] = 1  # cycle through the inputs one by one

    h5data = ModifiedH5Loader(config, config["model"]["num_bins"], config["model"]["round_encoding"])
    dataloader = create_dataloader(h5data, config, config_parser)
    init_dataloader = create_dataloader(h5data, config, config_parser)  # create second dataloader so they start at the same index

    data = []
    for i in range(len(h5data.files)): 
        file = h5py.File(h5data.files[i], 'r')
        data += [np.vstack((
            file['ground_truth/timestamp'],
            file['ground_truth/tx'], 
            file['ground_truth/ty'], 
            file['ground_truth/tz'], 
            file['ground_truth/qx'], 
            file['ground_truth/qy'], 
            file['ground_truth/qz'], 
            file['ground_truth/qw'], 
        )).T]
    data = np.vstack(data)
    data[:, 0] -= h5data.open_files[0].attrs['gt0']

    start_times, end_times = h5data.get_start_end_times()
    print("Using data from ", data_dir, ': \n',
          len(start_times), "files: ", start_times, end_times)  
    #      '(', round(end_time - start_time, 3), 's)')
    print("Range of numpy data: ", data[0,0], "-", data[-1,0])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sys.modules['new_files.laurens_stuff'] = sys.modules.get('new_files.rotation_utils', None)  # deal with file renaming
    model = torch.load(model_file, map_location=device)
    if not 'include_init' in dir(model): model.include_init = False
    model.device = device
    model.eval()

    init_time = 0  # TODO fix rotation in plot to remove the take-off phase
    init_time, init_pos, init_rot, rot_offset = find_init_data(data, init_dataloader, init_time)

    data -= np.array([init_time] + list(init_pos) + [0, 0, 0, 0])

    fig, ax, model_ax, im_ax, ev_ax, imu_axs, limits, spread = initialize_plots(data, images) 
    
    orienter, true_prev_rot, pred_prev_rot = initialize_rotation(data, spread, init_pos, init_rot, prev_rot=True)
    #orienter, true_prev_rot, pred_prev_rot = initialize_rotation(init_dataloader, spread, init_pos, init_rot, prev_rot=True)

    return dataloader, h5data, data, fig, ax, limits, ev_ax, im_ax, imgs, imu_axs, accs, orienter, init_rot, rot_offset, start_times[0], spread, model_ax, model, true_prev_rot, pred_prev_rot

    #data = np.genfromtxt(data_dir + '/' + ground_truth_file, delimiter=' ', skip_header=1)
    
    init_idx = 0  # TODO fix rotation in plot, then set to smth like 2000, should remove the take-off phase
    start_time, end_time = float(data[init_idx, ground_truth_cols.index('t')]), float(data[-1, ground_truth_cols.index('t')])
    print(start_time - data[0, ground_truth_cols.index('t')]) 

    # set the initial time to 0
    data[:, ground_truth_cols.index('t')] = np.round( data[:, ground_truth_cols.index('t')] - start_time , 3)
    
    # set the initial position to 0
    x, y, z = data[init_idx][ground_truth_cols.index('x')], data[init_idx][ground_truth_cols.index('y')], data[init_idx][ground_truth_cols.index('z')]
    arr = np.zeros(data.shape[1])
    arr[ground_truth_cols.index('x')] = x
    arr[ground_truth_cols.index('y')] = y
    arr[ground_truth_cols.index('z')] = z
    data = data - arr

    fig, ax, model_ax, im_ax, ev_ax, imu_axs, limits, spread = initialize_plots(data, images)

    model = torch.load(model_file, map_location=torch.device('cpu'))
    model.eval()
    
    # if rotation:
    orienter, init_rot, true_prev_rot, pred_prev_rot = initialize_rotation(data, spread, prev_rot=True, idx=init_idx)

    # if events:
    initialize_events(data)

    # if images:
    #     imgs = initialize_images(start_time, data_dir, image_file, image_cols)

    # if imu:
    #     accs = initialize_imu(start_time, data_dir, imu_file, imu_cols)

    print('Using data from ', data_dir,
          '(', round(end_time - start_time, 3), 's)')
    return data, fig, ax, limits, ev_ax, im_ax, imgs, imu_axs, accs, orienter, init_rot, start_time, spread, model_ax, model, true_prev_rot, pred_prev_rot


def create_dataloader(h5data, config, config_parser):
    return torch.utils.data.DataLoader(
        h5data,
        drop_last=True,
        batch_size=config["loader"]["batch_size"],
        collate_fn=h5data.custom_collate,
        worker_init_fn=config_parser.worker_init_fn,
        **config_parser.loader_kwargs,
    )


def find_init_data(data, dataloader, init_time):

    for idx, d in enumerate(dataloader):
        if idx == 0 or (d['gt_time'].item() <= init_time <= d['gt_time'].item() + d['gt_dt'].item()): 
            #first_row = d
            init_time = d['gt_time'].item()
            break
    
    init_pos = data[data[:, 0] == init_time][0, 1:4]
    
    #init_rot = R.from_quat(data[data[:, 0] == init_time][0, 4:])
    #init_rot = R.from_matrix(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])) 
    init_rot = find_orientation(data, init_pos, noise_margin) 
    rot_offset = R.from_quat(data[data[:, 0] == init_time][0, 4:]).inv() * init_rot 

    return init_time, init_pos, init_rot, rot_offset

#     return first_row['gt_time'].item(), first_row['gt_translation'].cpu().detach().numpy().reshape(-1), R.from_rotvec(first_row['gt_rotation'].cpu().detach().numpy().reshape(-1))


def initialize_plots(data, images):  # adapted from the other file
    fig = plt.figure()
    im_ax, ev_ax, imu_axs = None, None, None

    ax = fig.add_subplot(121, projection='3d')
    model_ax = fig.add_subplot(122, projection='3d')

    xyz = np.vstack((data[:, 1], data[:, 2], data[:, 3]))
    limits = np.vstack((np.min(xyz, axis=1), np.max(xyz, axis=1), (np.max(xyz, axis=1)+np.min(xyz, axis=1))/2, (np.max(xyz, axis=1)-np.min(xyz, axis=1))))
    #limits = np.array([[-10, -10, -1], [10, 10, 1]])  # TODO adjust
    #limits = np.vstack((limits, (limits[1]+limits[0])/2, (limits[1]-limits[0])/2))

    spread = 1.1 * max(limits[3, :])

    round_to = 2  # significant figures
    round_to = round_to - int(np.floor(np.log10(abs(spread)))) - 1

    limits = np.round(np.vstack((limits[2, :] - spread/2, limits[2, :] + spread/2)), round_to).T
    limits = np.vstack((limits, np.array([[0, 360], [0, 270]])))  # TODO tune automatically

    return fig, ax, model_ax, im_ax, ev_ax, imu_axs, limits, spread


# def initialize_events(evts):  # a copy from the other file
#     global event_cols, event_window, dt, ev_idx

#     if 'p' not in event_cols:
#         evts = np.hstack((evts, np.ones((len(evts), 1))))
#         event_cols += ['p']

#     if not event_window or event_window > dt:
#         event_window = dt
#     print('event_window = ', event_window)

#     ev_idx = 0


def initialize_rotation(data, spread, init_pos, init_rot, prev_rot=False):  # adapted from the other file
    #orienter = make_arrow(spread)
    orienter = make_drone(spread)

    # set initial rotation to zero
    #orienter = init_rot.inv().apply(orienter)  # compensation
    # the above pre-rotates the arrow to compensate for the initial rotation offset

    ## determine initial orientation
    #init_rot = find_orientation(data, init_pos, noise_margin)  

    if prev_rot:
        true_prev_rot = pred_prev_rot = init_rot
        return orienter, true_prev_rot, pred_prev_rot
    else:
        return orienter


def find_orientation(data, init_pos, noise_margin):  # modified from a copy from the other file
    for idx, row in enumerate(data):
        x, y = row[1], row[2]
        if np.sqrt((x - init_pos[0])**2 + (y - init_pos[1])**2) >= noise_margin:
            break

    # find the initial direction in x and y
    init_dir = np.array([x - init_pos[0], y - init_pos[1], 0])
    x, y = init_dir[0] / np.linalg.norm(init_dir), init_dir[1] / np.linalg.norm(init_dir)

    # return corresponding rotation matrix assuming rotation about z-axis
    return R.from_matrix(np.array([
        [x, -y, 0],
        [y, x, 0],
        [0, 0, 1]
    ]))


def show_orientation(rot, show=True):
    rot_matrix = rot.as_matrix()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = ['r', 'g', 'b']
    for column, c in zip(rot_matrix.T, colors):
        column *= 0.05  # rescale to fit the plot limits
        ax.quiver(
            0, 0, 0,
            column[0], column[1], column[2], 
            color=c
        )
    if show: plt.show()


# updates
def update(inp, data, fig, ax, limits, ev_ax, im_ax, imgs, imu_axs, accs, orienter, init_rot, rot_offset, spread, model_ax, model):  # adapted from the other file
    global true_prev_rot, pred_prev_rot

    # prev_time = inp['gt_time'].item()
    # prev_idx = np.where(data[:,0] == prev_time)[0][0]
    current_time = inp['gt_time'].item() + inp['gt_dt'].item()  # latest timestamp
    try: 
        current_idx = np.where(data[:,0] == current_time)[0][0]  # index of latest gt entry
    except IndexError: 
        print("IndexError: can't find current_time in data")
        print(current_time, np.where(data[:,0] == current_time)[0], data[-1,0])
        current_idx = -1
    current_pos = data[current_idx,1:4]

    #x, y, z = data[i][ground_truth_cols.index('x')], data[i][ground_truth_cols.index('y')], data[i][ground_truth_cols.index('z')]
    x, y, z = current_pos
    ax = prepare_ax(data, current_idx, ax, spread, zoom, x, y, z)
    model_ax = prepare_ax(data, current_idx, model_ax, spread, zoom, x, y, z)

    # if rotation:
    rot = update_rotation(data, current_idx) 
    rot = rot*rot_offset  # now x axis is flight direction, y-axis is left, z-axis is up
    # show_orientation(rot_offset, show=False)
    # show_orientation(rot*rot_offset)
    # assert False
    true_diff = rot*true_prev_rot.inv()
    true_prev_rot = rot  # store for next iteration
    ax = plot_rotation(rot, orienter, ax, x, y, z)  # update plot
    if not save:
        print("\nActual orientation: ", rot.as_rotvec(), "actual difference: ", true_diff.as_rotvec())  # log values

    target_diff = inp['gt_rotation'].numpy()
    rot = update_model_rotation(target_diff, 'difference', model.rotation_type)
    target_diff = rot*pred_prev_rot.inv()
    if not save:
        print("Target orientation: ", rot.as_rotvec(), "target difference: ", target_diff.as_rotvec())  #  log values

    # # if events:
    # inp_cnt = get_event_cnt(t)
    # print(torch.sum(inp_cnt).item())  # TODO remove
    # dummy_voxel = torch.randn((8, 2, 128, 128), dtype=torch.float32)

    with torch.no_grad():
        #rot = model(dummy_voxel, inp_cnt)['rotation'].numpy()
        rot = model(inp['event_voxel'].to('cpu'), inp['event_cnt'].to('cpu'))['rotation'].numpy()
    rot = update_model_rotation(rot, 'difference', model.rotation_type)
    pred_diff = rot*pred_prev_rot.inv()
    pred_prev_rot = rot # store for next iteration
    #pred_prev_rot = true_prev_rot
    model_ax = plot_rotation(rot, orienter, model_ax, x, y, z)  # update plot
    if not save:
        print("Predicted orientation: ", rot.as_rotvec(), "predicted difference: ", pred_diff.as_rotvec())  # log values
    
    if not save: 
        print("MSE gt: ", np.mean((np.array(true_diff.as_rotvec()).reshape(-1) - np.array(pred_diff.as_rotvec()).reshape(-1))**2))
        print("MSE target: ", np.mean((np.array(target_diff.as_rotvec()).reshape(-1) - np.array(pred_diff.as_rotvec()).reshape(-1))**2), '\n')

    # if images:
    #     im_ax = update_image(t, dt, im_ax, imgs)

    # if imu:
    #     imu_axs = update_imu(i, imu_axs, accs)

    fig.tight_layout()
    img = np.array(fig.canvas.renderer.buffer_rgba())
    img = img[:, :, :3]
    return img


def prepare_ax(data, i, ax, spread, zoom, x, y, z):
    ax.clear()

    # plot the position
    ax.scatter(x, y, z, label='current location', c='b', marker='o')
    # plot the trajectory
    ax.plot(data[:i+1, 1], data[:i+1, 2], data[:i+1, 3], label='trajectory', c='b')

    if focus:
        limits = np.array([
            [x - 1.1 * spread / (zoom/10), x + 1.1 * spread / (zoom/10)],
            [y - 1.1 * spread / (zoom/10), y + 1.1 * spread / (zoom/10)],
            [z - 1.1 * spread / (zoom/10), z + 1.1 * spread / (zoom/10)],
        ])
    ax.set_xlim(limits[0][0], limits[0][1])
    ax.set_ylim(limits[1][0], limits[1][1])
    ax.set_zlim(limits[2][0], limits[2][1])

    return ax


# def update_events(t, ev_ax):  # adapted from the main file
#     global event_window, ev_idx, start_time
#     print(t)  # TODO remove
#     # ev_ax.clear()

#     e = []
#     with open(data_dir + '/' + event_file, 'r') as file:
#         if ev_idx:
#             file.seek(ev_idx)
#         else:
#             ev_idx = 209598478
#             file.seek(ev_idx)
#         old_ev_idx = ev_idx  # TODO remove
#         line = file.readline()
#         ev_stop = False
#         while not ev_stop:
#             values = line.split()
#             if float(values[event_cols.index('t')]) >= t + start_time:
#                 values[event_cols.index('y')] = 270 - float(values[event_cols.index('y')])  # TODO tune automatically
#                 e += [values]
#             line = file.readline()
#             if float(values[event_cols.index('t')]) >= t + event_window + start_time or line is None:
#                 ev_stop = True
#                 ev_idx = file.tell()
#         print("# events: ", ev_idx - old_ev_idx)  # TODO remove
#     e = np.array(e, dtype=float)
#     return e


# def get_event_cnt(t):  # adapted from Jesse's code
#     img_size = [271, 360]
#     inp_size = [128, 128]  # TODO: adjust this automatically
#     e = update_events(t, ev_ax=None)

#     ps = torch.tensor(e[:, event_cols.index('p')], dtype=torch.float32)
#     xs = torch.tensor(e[:, event_cols.index('x')], dtype=torch.long)
#     ys = torch.tensor(e[:, event_cols.index('y')], dtype=torch.long)
    
#     mask = ps.clone()
#     mask[ps < 0] = 0
    
#     pos_cnt = torch.zeros(img_size, dtype=torch.float32)
#     pos_cnt.index_put_((ys, xs), ps*mask, accumulate=True)
#     pos_cnt = transforms.functional.resize(pos_cnt.unsqueeze(0), size=inp_size).squeeze(0)

#     mask = ps.clone()
#     mask[ps > 0] = 0
    
#     neg_cnt = torch.zeros(img_size, dtype=torch.float32)
#     neg_cnt.index_put_((ys, xs), ps*mask, accumulate=True)
#     neg_cnt = transforms.functional.resize(neg_cnt.unsqueeze(0), size=inp_size).squeeze(0)

#     return torch.stack([pos_cnt, neg_cnt]).unsqueeze(0)


def update_rotation(data, i):  # adapted from the other file
    return R.from_quat(data[i][4:])
    

def plot_rotation(rot, arrow, ax, x, y, z):  # adapted from the other file

    # add arrows to make orientation more clear
    rot_matrix = rot.as_matrix()
    colors = ['r', 'g', 'b']
    for column, c in zip(rot_matrix.T, colors):
        column *= 0.5  # rescale to fit the plot limits
        ax.quiver(
            x, y, z,
            column[0], column[1], column[2], 
            color=c
        )

    rotated_obj = rot.apply(arrow)
    #rotated_obj = init_rot.apply(rotated_obj)
    translated_obj = rotated_obj + np.array([x, y, z])

    ax.plot(translated_obj[:, 0], translated_obj[:, 1], translated_obj[:, 2], c='black')
    return ax


def update_model_rotation(r, mode, typ):  # new
    if typ == 'quat':
        r = R.from_quat(r)
    elif typ == 'rotvec':
        r = R.from_rotvec(r)
    elif typ == 'matrix':
        r = R.from_matrix(np.reshape(r, (3, 3)))
    elif typ == 'euler':
        r = R.from_euler('xyz', r, degrees=False)
    elif typ == 'euler_deg':
        r = R.from_euler('xyz', r, degrees=True)
    else: 
        raise ValueError("Unknown rotation type")
    
    if mode == 'absolute':
        return r
    elif mode == 'difference':
        global pred_prev_rot
        if r.magnitude == 0: 
            return pred_prev_rot
        else:
            r = r*pred_prev_rot
            return r

if __name__ == '__main__':
    # main loop
    dataloader, h5data, data, fig, ax, limits, ev_ax, im_ax, imgs, imu_axs, accs, orienter, init_rot, rot_offset, start_time, spread, model_ax, model, true_prev_rot, pred_prev_rot = initialize()

    if save:
        #TODO: fix frame rate somehow
        frame_rate = int(speed * 1/dt)  # frames per second
        frames = []
        #t = 0
        end = data[-1, 0]
        #while t <= end:
        
        for inp in dataloader:

            frame = update(inp, data, fig, ax, limits, ev_ax, im_ax, imgs, imu_axs, accs, orienter, init_rot, rot_offset, spread, model_ax, model)

            frames += [frame]

            t = inp['gt_time'].item()
            dt = inp['gt_dt'].item()

            #i = np.where(np.round(data[:, ground_truth_cols.index('t')] / dt) == round(t / dt))[0][0]
            #frames += [update(i, t, data, fig, ax, limits, ev_ax, im_ax, imgs, imu_axs, accs, orienter, init_rot, spread, model_ax, model)]

            print_progress(t/end)
            plt.pause(0.001)
            # plt.pause(dt)
            #t += dt

            if t >= 10: break  # keep the video short

        frames = frames[1:]
        imageio.mimsave(file_to_write, frames, 'mp4', fps=frame_rate)

    else:
        for inp in dataloader:

            update(inp, data, fig, ax, limits, ev_ax, im_ax, imgs, imu_axs, accs, orienter, init_rot, rot_offset, spread, model_ax, model)

            t = inp['gt_time'].item()
            dt = inp['gt_dt'].item()

            plt.pause(dt)

            # plt.show()
            # assert False

            if h5data.new_seq: break

        # t = 0
        # end = data[-1, ground_truth_cols.index('t')]
        # while t <= end:
        #     try:
        #         i = np.where(np.round(data[:, 0]/dt) == round(t/dt))[0][0]
        #         update(i, t, data, fig, ax, limits, ev_ax, im_ax, imgs, imu_axs, accs, orienter, init_rot, spread, model_ax, model)

        #     except Exception as e:
        #         print('Could not find entry at time ', t, '\n', e)

        #     plt.pause(dt)
        #     t += dt

    print('Done.', flush=True)
