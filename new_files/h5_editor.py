"""
WORK IN PROGRESS
"""

# imports
import sys, os
import shutil
import h5py
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from event_flow_pipeline.dataloader.encodings import binary_search_array
from event_flow_pipeline.dataloader.h5 import Frames, FlowMaps


def print_structure(name, obj):
    print(name)


def inspect_hd5(data_dir, file_to_read):  # inspecting the files
    with h5py.File(data_dir + '/' + file_to_read, 'r') as file:
        print("Keys:", file.keys())

        # printing the structure
        file.visititems(print_structure)
        print(list(file.attrs))

        print(file.attrs['t0'])
        print(file["events/ts"][-1])
        print(file["events/ts"][-1] - file["events/ts"][0])
        print(min(file["events/xs"]), max(file["events/xs"]))
        print(min(file["events/ys"]), max(file["events/ys"]))
        
        # # Getting the data
        # data = list(file[a_group_key])
        # print(data)

        # print(file['events']['xs'])
        # print(file['ground_truth']['tx'])
        
        # plt.imshow(file['images']['image000000010'][:,:], cmap='gray')
        # plt.show()

# okay, so the h5 file consists of images and event data;
# the event data has label 'events' and has 4 subgroups: 'ps' (polarity, +1 or -1), 'ts' (time in s?), 'xs' (horizontal pixel position, 0-127), 'ys' (vertical pixel position, 0-127);
# each column has shape (500000, ) and type "|b1" (bool, ps) or "<f8" (float, ts) or "<i2" (integer, other columns); 
# the image data has label 'images' and has 18 subgroups: 'image000000000' up to 'image000000017' (1E9 digits); 
# each image has shape (128, 128) and type "|u1" (it is a grayscale image).


def find_ts_index(file, timestamp):
    """
    This function was copied from dataloader.h5.py.
    Find closest event index for a given timestamp through binary search.
    """

    return binary_search_array(file["events/ts"], timestamp)
    

def get_event_index(mode, window, file, file_frames, file_flowmaps, batch_row):
    """
    This function was copied from dataloader.h5.py.
    Get all the event indices to be used for reading.
    :param batch: batch index
    :param window: input window
    :return event_idx: event index
    """
    event_idx0 = None
    event_idx1 = None
    if mode == "events":
        event_idx0 = batch_row
        event_idx1 = batch_row + int(window)  # change
    elif mode == "time":
        event_idx0 = find_ts_index(
            file, batch_row + file.attrs["t0"]
        )
        event_idx1 = find_ts_index(
            file, batch_row + file.attrs["t0"] + window
        )
    elif mode == "frames":
        idx0 = int(np.floor(batch_row))
        idx1 = int(np.ceil(batch_row + window))
        if window < 1.0 and idx1 - idx0 > 1:
            idx0 += idx1 - idx0 - 1
        event_idx0 = find_ts_index(file, file_frames.ts[idx0])
        event_idx1 = find_ts_index(file, file_frames.ts[idx1])
    elif mode == "gtflow_dt1" or mode == "gtflow_dt4":
        idx0 = int(np.floor(batch_row))
        idx1 = int(np.ceil(batch_row + window))
        if window < 1.0 and idx1 - idx0 > 1:
            idx0 += idx1 - idx0 - 1
        event_idx0 = find_ts_index(file, file_flowmaps.ts[idx0])
        event_idx1 = find_ts_index(file, file_flowmaps.ts[idx1])
    else:
        print("DataLoader error: Unknown mode.")
        raise AttributeError
    return event_idx0, event_idx1


def inspect_windows(file_path, mode=str, window=float, verbose=True):

    file = h5py.File(file_path, 'r') # reading the file

    file_frames = Frames()
    file["images"].visititems(file_frames)

    file_flowmaps = FlowMaps()
    if mode == "gtflow_dt1":
        file["flow_dt1"].visititems(file_flowmaps)
    elif mode == "gtflow_dt4":
        file["flow_dt4"].visititems(file_flowmaps)

    batch_row = 0
    n_events = []
    d_time = []

    while True:
        try:
            idx0, idx1 = get_event_index(mode, window, file, file_frames, file_flowmaps, batch_row)
            n_events += [idx1 - idx0]
            d_time += [file["events/ts"][idx1] - file["events/ts"][idx0]]
            batch_row += window
        except IndexError as e:
            if verbose: print("IndexError: ", e)
            break
    
    if verbose:
        print(f"Min, mean & max number of events: {min(n_events), np.mean(n_events), max(n_events)}")
        print(f"Min, mean & max time difference: {min(d_time), np.mean(d_time), max(d_time)}")
    return min(n_events), np.mean(n_events), max(n_events), min(d_time), np.mean(d_time), max(d_time)


def convert_events_to_hd5(data_dir, events_file, file_to_write):  # for attempting my own conversion

    data = np.genfromtxt(data_dir + '/' + events_file, delimiter=' ', skip_header=1)  # loading the data
    columns = ['ps', 'ts', 'xs', 'ys']
    typs = [bool, float, np.int16, np.int16]

    with h5py.File(data_dir + '/' + file_to_write, 'a') as file:  # editing the file
        events_group = file.create_group('events')
        for i in range(data.shape[1]):
            # TODO: next line hasn't been tested!
            events_group.create_dataset(columns[i], data=data[:50000, i].astype(typs[i]))  # creating a subgroup for each column, use the correct type


def convert_groundtruth_to_hd5(data_dir, ground_truth_file, file_to_write, low=0, high=50000, add_init=False):
    
    data = np.genfromtxt(data_dir + '/' + ground_truth_file, delimiter=' ', skip_header=1)  # loading the data
    columns = ['timestamp', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']

    with h5py.File(data_dir + '/' + file_to_write, 'a') as file:  # editing the file
        gt_group = file.create_group('ground_truth')
        for i in range(data.shape[1]):
            gt_group.create_dataset(columns[i], data=data[low:high, i].astype(float))  # creating a subgroup for each column
        
        if add_init:
            file.attrs['gt0'] = data[columns.index('timestamp'), 0]  


def modify_existing(h5_data_dir, txt_data_dir, partial_name, ground_truth_file='groundtruth.txt'):
    ts = np.genfromtxt(txt_data_dir + '/' + ground_truth_file, delimiter=' ', skip_header=1)[:, 0]

    files = os.listdir(h5_data_dir)
    files = list(filter(lambda file: partial_name in file, files))
    for i in range(len(files)):
    
        file_to_read = partial_name + '_' + str(i) + '.h5'
        assert file_to_read in files
        file_to_cache = partial_name[:-2] + 'temp_' + str(i) + '.h5'
        file_to_write = partial_name[:-2] + 'rotation_' + str(i) + '.h5'
        shutil.copy(h5_data_dir + '/' + file_to_read, txt_data_dir + '/' + file_to_cache)

        with h5py.File(h5_data_dir + '/' + file_to_read, 'r') as original_file: 
            if original_file["events/ts"][-1] > ts[0] and original_file.attrs["t0"] < ts[-1]:
                low = np.where((original_file.attrs["t0"] < ts) & (ts < original_file["events/ts"][-1]))[0][0]
                high = np.where((original_file.attrs["t0"] < ts) & (ts < original_file["events/ts"][-1]))[0][-1]
            
                if (ts[high] - ts[low]) > 0.9 * (original_file["events/ts"][-1] - original_file.attrs["t0"]):
                    shutil.copy(txt_data_dir + '/' + file_to_cache, txt_data_dir + '/' + file_to_write)

                    convert_groundtruth_to_hd5(txt_data_dir, ground_truth_file, file_to_write, low, high, add_init=True)
            
        os.remove(txt_data_dir + '/' + file_to_cache)


# with h5py.File(data_dir + '/' + file_to_write, 'a') as file:
    #     for col, typ in zip(['ps', 'ts', 'xs', 'ys'], [bool, float, np.int16, np.int16]):
    #         data = file['events'][col][:].astype(typ)
    #         del file['events'][col]
    #         file['events'].create_dataset(col, data=data)


# Notes on integrating in existing files: 
# - use indoor_forward_3_davis_with_gt_3 up to ..._10
# - split according to ['events/ts'][-1] and attrs['t0']


def convert_files():
    new_data_dir = 'datasets/data/rotation_demo'
    h5_data_dir = 'datasets/data/training'
    txt_data_dir = 'datasets/data/txt/'
    events_file = 'events.txt'
    ground_truth_file = 'groundtruth.txt'
    #file_to_write = 'test.h5'  # 'indoor_forward_3_davis_with_gt_0.h5'
    
    #convert_events_to_hd5(data_dir, events_file, file_to_write)  # takes ~15 mins
    #convert_groundtruth_to_hd5(data_dir, ground_truth_file, file_to_write)
    #inspect_hd5(h5_data_dir, 'indoor_forward_3_davis_with_gt_3.h5')
    for n in [5, 7, 9, 10]:
        partial_name = f'indoor_forward_{n}_davis_with_gt'
        txt_data_dir_ = txt_data_dir + partial_name
        modify_existing(h5_data_dir, txt_data_dir_, partial_name=partial_name)

    print('Done.')


if __name__ == "__main__":
    window = 0.001  # first 0.001, then 0.002 and finally 0.005
    idxs = [8, 11, 12]  # [3, 5, 6, 7, 9, 10]
    mins = [0, 0, 0]  # [0, 0, 0, 0, 0, 0]
    maxs = [23, 17, 11]  # [12, 19, 10, 17, 12, 8]
    minis = means = maxis = []
    for idx, i in enumerate(idxs):
        for j in range(mins[idx], maxs[idx]+1):
            mini, mean, maxi, _, _, _ = inspect_windows(
                #file_path = f"datasets/data/UZH-FPV/flow/indoor_forward_{i}_davis_with_gt_{j}.h5", 
                file_path = f"datasets/data/UZH-FPV/flow/indoor_forward_{i}_davis_{j}.h5", 
                mode="time",  # events / time / frames / gtflow_dt1 / gtflow_dt4
                window=window,
                verbose=False
                )
            minis += [mini]
            means += [mean]
            maxis += [maxi]
        print(i, "done")
    print(len(maxis), max(maxis))
