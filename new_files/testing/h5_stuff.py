"""
WORK IN PROGRESS
"""

# imports
import os
import shutil
import h5py
import numpy as np
import matplotlib.pyplot as plt

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

    for i in range(13):
    
        file_to_read = partial_name + '_' + str(i) + '.h5'
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


if __name__ == '__main__':
    new_data_dir = 'datasets/data/rotation_demo'
    h5_data_dir = 'datasets/data/training'
    txt_data_dir = 'datasets/data/txt/indoor_forward_3_davis_with_gt'
    events_file = 'events.txt'
    ground_truth_file = 'groundtruth.txt'
    #file_to_write = 'test.h5'  # 'indoor_forward_3_davis_with_gt_0.h5'
    
    #convert_events_to_hd5(data_dir, events_file, file_to_write)  # takes ~15 mins
    #convert_groundtruth_to_hd5(data_dir, ground_truth_file, file_to_write)
    #inspect_hd5(h5_data_dir, 'indoor_forward_3_davis_with_gt_3.h5')
    modify_existing(h5_data_dir, txt_data_dir, partial_name='indoor_forward_3_davis_with_gt')

    print('Done.')
