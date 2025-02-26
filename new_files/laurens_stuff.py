"""
Based on Jesse's file
"""
# imports
import torch
import numpy as np
from scipy.spatial.transform import Rotation

from event_flow_pipeline.models.base import BaseModel
from event_flow_pipeline.models.model_util import copy_states, CropParameters
from event_flow_pipeline.loss.flow import BaseValidationLoss
from event_flow_pipeline.dataloader.h5 import H5Loader, Frames, ProgressBar, FlowMaps
import h5py


# first attempt
class SimpleRotationModel(BaseModel):
    # normal ANN
    # TODO: make easier to modify
    # TODO: implement spiking version

    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, n_outputs)
        )
        
    def forward(self, X):
        return self.layers(X)
    

# second attempt
class ConvRotationModel(BaseModel):
    # CNN
    # TODO: make easier to modify
    # TODO: implement spiking version

    def __init__(self, input_size, n_outputs):
        super().__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(input_size[1], 16, kernel_size=3, stride=2, padding=1),  # 128 -> 64
            torch.nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            torch.nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1),  # 32 -> 16
        )
        n_middle = self.conv_layers(torch.randn(input_size)).reshape(input_size[0], -1).shape[1]
        self.linear_layers = torch.nn.Sequential(
            torch.nn.Linear(n_middle, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, n_outputs)
        )
        self.layers = torch.nn.Sequential(
            self.conv_layers, 
            torch.nn.Flatten(),
            self.linear_layers
        )
    
    def forward(self, X):
        return self.layers(X)
    

# integrating everything
class FullRotationModel(BaseModel):
    # the following was adapted from the FireNet code

    def __init__(self, unet_kwargs):
        super().__init__()

        # self.num_bins = unet_kwargs["num_bins"]
        # base_num_channels = unet_kwargs["base_num_channels"]
        # kernel_size = unet_kwargs["kernel_size"]
        # self.encoding = unet_kwargs["encoding"]
        # self.norm_input = False if "norm_input" not in unet_kwargs.keys() else unet_kwargs["norm_input"]
        self.mask = unet_kwargs["mask_output"]
        # ff_act, rec_act = unet_kwargs["activations"]
        # if type(unet_kwargs["spiking_neuron"]) is dict:
        #     for kwargs in self.kwargs:
        #         kwargs.update(unet_kwargs["spiking_neuron"])
        self.device = unet_kwargs["device"]

        self.flow_model = unet_kwargs["flow_model"]
        self.flow_model.eval()

        self.input_size = tuple([unet_kwargs["batch_size"]] + [2] + list(unet_kwargs["resolution"]))
        
        self.transfer_size = None
        self.n_transfers = None
        self.current_transfer = None
        self.transfer_layer = unet_kwargs["transfer_layer"]
        self.get_n_transfers()  # will update self.transfer_size and self.n_transfers

        self.rotation_mode = unet_kwargs["rotation_mode"]
        self.rotation_type = unet_kwargs["rotation_type"]
        self.n_outputs = self.get_n_outputs()

        if unet_kwargs["model_type"] == ('conv' or 'conv_model' or 'ConvRotationModel'):
            self.model_type = 'conv'
            self.rotation_model = ConvRotationModel(input_size=list(self.transfer_size), n_outputs=self.n_outputs)
        else:
            self.model_type = 'linear'
            self.rotation_model = SimpleRotationModel(n_inputs=self.n_transfers, n_outputs=self.n_outputs)
    
    def transfer_hook(self, module, input, output):  # should I use output[0] or not?
        if self.transfer_size is None: 
            self.transfer_size = output.shape
        if self.n_transfers is None: 
            self.n_transfers = len(torch.flatten(output[0]))
        self.current_transfer = output
    
    def get_n_transfers(self):
        # first, set the hook on the transfer layer
        hook = getattr(self.flow_model, self.transfer_layer).register_forward_hook(self.transfer_hook)

        # now pass a dummy input (n_transfers will be registered)
        dummy_voxel = torch.randn(self.input_size, dtype=torch.float32).to(self.device)
        dummy_cnt = torch.randint(0, 3, self.input_size, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outp = self.flow_model(dummy_voxel, dummy_cnt)

        hook.remove()
        return self.n_transfers
    
    def get_n_outputs(self):
        if self.rotation_type == "rotvec" or self.rotation_type == "euler":
            return 3
        elif self.rotation_type == "quat":
            return 4
        elif self.rotation_type == "matrix":
            return 9

    # @property
    # def states(self):
    #     return copy_states(self._states)

    # @states.setter
    # def states(self, states):
    #     self._states = states

    def detach_states(self):
        self.flow_model.detach_states()
    #     detached_states = []
    #     for state in self.states:
    #         if type(state) is tuple:
    #             tmp = []
    #             for hidden in state:
    #                 tmp.append(hidden.detach())
    #             detached_states.append(tuple(tmp))
    #         else:
    #             detached_states.append(state.detach())
    #     self.states = detached_states

    def reset_states(self):
        self.flow_model.reset_states()
    #     self._states = None

    # def init_cropping(self, width, height):
    #     pass

    def forward(self, event_voxel, event_cnt, log=False):
        """
        :param event_voxel: N x num_bins x H x W
        :param event_cnt: N x 2 x H x W per-polarity event cnt and average timestamp
        :param log: log activity
        :return: output dict with list of [N x 2 X H X W] (x, y) displacement within event_tensor.
        """
        event_cnt = event_cnt.to(self.device)
        event_voxel = event_voxel.to(self.device)   

        # forward pass
        hook = getattr(self.flow_model, self.transfer_layer).register_forward_hook(self.transfer_hook)
        
        with torch.no_grad():
            flow = self.flow_model(event_voxel, event_cnt, log=log)['flow'][0]
        
        if self.model_type == 'linear':
            transfer = torch.reshape(self.current_transfer, (self.current_transfer.shape[0], self.n_transfers))
        else: 
            transfer = self.current_transfer

        r = self.rotation_model(transfer)
        
        hook.remove()

        # TODO: potentially log activity
        activity = None

        return {"flow": [flow], "activity": activity, "rotation": r}


class RotationLoss(BaseValidationLoss):

    def __init__(self, config, device, flow_scaling=128):
        super().__init__(config, device, flow_scaling)
        self.loss_fn = torch.nn.MSELoss()  
        self.y_test = None
        self.y_pred = None

    def prepare_loss(self, y_pred, y_test):
        self.y_pred = y_pred
        self.y_test = y_test
        # print(y_pred)
        # print(y_test)
        # print()

    def forward(self):
        return self.loss_fn(self.y_pred, self.y_test)


class ModifiedH5Loader(H5Loader):
    
    def __init__(self, config, num_bins, round_encoding=False):
        super().__init__(config, num_bins, round_encoding)
        self.rotation_mode = config["loader"]["rotation_mode"]
        self.rotation_type = config["loader"]["rotation_type"]

        # self.original_getitem = super().__getitem__
    
    def get_start_end_times(self):
        """
        This function returns 2 tuples, start_time and end_time, resp. the first and last timestamp in each file's ground_truth data.
        """
        start_times, end_times = [], []
        for file_path in self.files:
            with h5py.File(file_path, 'r') as file:
                start_times += [file['ground_truth/timestamp'][0] - file.attrs['gt0']]
                end_times += [file['ground_truth/timestamp'][-1] - file.attrs['gt0']]
        return start_times, end_times

    def get_gt_index(self, file, t1, t2):
        if t2 < file["ground_truth/timestamp"][0] - file.attrs["t0"]:
            # if the interval occurs before the gt data, simply provide the first row
            idx1 = 0
            idx2 = 1
        elif file["ground_truth/timestamp"][-1] - file.attrs["t0"] < t1:
            # if the interval occurs after the gt data, simply provide the last row
            idx1 = -1
            idx2 = None
        else: 
            idxs = np.where((t1 < (file["ground_truth/timestamp"] - file.attrs["t0"])) 
                    & ((file["ground_truth/timestamp"] - file.attrs["t0"]) < t2))[0]
            idx1 = idxs[0] - 1 if idxs[0]!= 0 else idxs[0]
            idx2 = idxs[-1] + 1 if idxs[-1] != len(file["ground_truth/timestamp"])-1 else idxs[-1]
        # TODO: implement error handling: what if torch.where is empty? 
        return idx1, idx2

    def get_translation_rotation(self, file, t1, t2):
        idx1, idx2 = self.get_gt_index(file, t1, t2)

        timestamps = file["ground_truth/timestamp"][idx1:idx2]
        tx = file["ground_truth/tx"][idx1:idx2]
        ty = file["ground_truth/ty"][idx1:idx2]
        tz = file["ground_truth/tz"][idx1:idx2]
        qx = file["ground_truth/qx"][idx1:idx2]
        qy = file["ground_truth/qy"][idx1:idx2]
        qz = file["ground_truth/qz"][idx1:idx2]
        qw = file["ground_truth/qw"][idx1:idx2] 

        timestamps = torch.tensor(timestamps)

        t = np.transpose(np.vstack((tx, ty, tz)))
        if self.rotation_mode == "absolute":
            t = np.mean(t, axis=0)  # TODO maybe this should also be t[-1]?
        elif self.rotation_mode == "difference":
            t = t[-1] - t[0]
        elif self.rotation_mode == "zero-offset":
            t = np.mean(t, axis=0) - file.attrs["gt0"][1:4]  # TODO maybe this should also be t[-1]?
        t = torch.flatten(torch.tensor(t, dtype=torch.float32))

        r = np.transpose(np.vstack((qx, qy, qz, qw)))
        if self.rotation_mode == "absolute":
            r = Rotation.from_quat(np.mean(r, axis=0))  # TODO maybe this should also be r[-1]?
        elif self.rotation_mode == "difference":
            r = Rotation.from_quat(r[-1]) * Rotation.from_quat(r[0]).inv()
        elif self.rotation_mode == "zero-offset":
            r = Rotation.from_quat(np.mean(r, axis=0)) * Rotation.from_quat(file.attrs["gt0"][4:]).inv()  # TODO maybe this should also be r[-1]?
        
        if self.rotation_type == "quat":
            r = r.as_quat()
        elif self.rotation_type == "rotvec":
            r = r.as_rotvec()
        elif self.rotation_type == "matrix":
            r = r.as_matrix()
        elif self.rotation_type == "euler":
            r = r.as_euler("xyz", degrees=False)
        
        r = torch.flatten(torch.tensor(r, dtype=torch.float32))
        
        if torch.isnan(t).any() or torch.isnan(r).any():
            raise ValueError(f"NaN value detected in ground truth: t1 = {t1}, t2 = {t2}, idx1 = {idx1}, idx2 = {idx2}, qx = {qx}, t = {r}")

        return timestamps, t, r

    def __getitem__(self, index):
        """
        Largely a copy of Jesse's funcion, but includes a get_translation_rotation function. 
        Changes are marked with a '# laurens' comment.
        """
        while True:
            batch = index % self.config["loader"]["batch_size"]

            # trigger sequence change
            len_frames = 0
            restart = False
            if self.config["data"]["mode"] == "frames":
                len_frames = len(self.open_files_frames[batch].ts)
            elif self.config["data"]["mode"] == "gtflow_dt1" or self.config["data"]["mode"] == "gtflow_dt4":
                len_frames = len(self.open_files_flowmaps[batch].ts)
            if (
                self.config["data"]["mode"] == "frames"
                or self.config["data"]["mode"] == "gtflow_dt1"
                or self.config["data"]["mode"] == "gtflow_dt4"
            ) and int(np.ceil(self.batch_row[batch] + self.config["data"]["window"])) >= len_frames:
                restart = True

            # load events
            xs = np.zeros((0))
            ys = np.zeros((0))
            ts = np.zeros((0))
            ps = np.zeros((0))

            if not restart:
                idx0, idx1 = self.get_event_index(batch, window=self.config["data"]["window"])

                if (
                    self.config["data"]["mode"] == "frames"
                    or self.config["data"]["mode"] == "gtflow_dt1"
                    or self.config["data"]["mode"] == "gtflow_dt4"
                ) and self.config["data"]["window"] < 1.0:
                    floor_row = int(np.floor(self.batch_row[batch]))
                    ceil_row = int(np.ceil(self.batch_row[batch] + self.config["data"]["window"]))
                    if ceil_row - floor_row > 1:
                        floor_row += ceil_row - floor_row - 1

                    idx0_change = self.batch_row[batch] - floor_row
                    idx1_change = self.batch_row[batch] + self.config["data"]["window"] - floor_row

                    delta_idx = idx1 - idx0
                    idx1 = int(idx0 + idx1_change * delta_idx)
                    idx0 = int(idx0 + idx0_change * delta_idx)

                xs, ys, ts, ps = self.get_events(self.open_files[batch], idx0, idx1)

                if ts.shape[0] > 0:  # laurens
                    gt_ts, t, r = self.get_translation_rotation(self.open_files[batch], ts[0], ts[-1])  # laurens
                    gt_dt = gt_ts[-1] - gt_ts[0]  # laurens
                    gt_time = gt_ts[0] - self.open_files[batch].attrs['gt0']  # laurens

            # trigger sequence change
            if (self.config["data"]["mode"] == "events" and xs.shape[0] < self.config["data"]["window"]) or (
                self.config["data"]["mode"] == "time"
                and self.batch_row[batch] + self.config["data"]["window"] >= self.batch_last_ts[batch]
            ):
                restart = True

            # handle case with very few events
            if xs.shape[0] <= 10:
                xs = np.empty([0])
                ys = np.empty([0])
                ts = np.empty([0])
                ps = np.empty([0])

                t = np.empty(3)  # laurens
                r = np.empty(3)  # laurens
                gt_dt = 0  # laurens
                gt_time = 0  # laurens  # TODO: check if this is the right approach

            # reset sequence if not enough input events
            if restart:
                self.new_seq = True
                self.reset_sequence(batch)
                self.batch_row[batch] = 0
                self.batch_idx[batch] = max(self.batch_idx) + 1

                self.open_files[batch].close()
                self.open_files[batch] = h5py.File(self.files[self.batch_idx[batch] % len(self.files)], "r")
                self.batch_last_ts[batch] = self.open_files[batch]["events/ts"][-1] - self.open_files[batch].attrs["t0"]

                if self.config["data"]["mode"] == "frames":
                    frames = Frames()
                    self.open_files[batch]["images"].visititems(frames)
                    self.open_files_frames[batch] = frames
                elif self.config["data"]["mode"] == "gtflow_dt1" or self.config["data"]["mode"] == "gtflow_dt4":
                    flowmaps = FlowMaps()
                    if self.config["data"]["mode"] == "gtflow_dt1":
                        self.open_files[batch]["flow_dt1"].visititems(flowmaps)
                    elif self.config["data"]["mode"] == "gtflow_dt4":
                        self.open_files[batch]["flow_dt4"].visititems(flowmaps)
                    self.open_files_flowmaps[batch] = flowmaps
                if self.config["vis"]["bars"]:
                    self.open_files_bar[batch].finish()
                    max_iters = self.get_iters(batch)
                    self.open_files_bar[batch] = ProgressBar(
                        self.files[self.batch_idx[batch] % len(self.files)].split("/")[-1], max=max_iters
                    )

                continue

            # event formatting and timestamp normalization
            dt_input = np.asarray(0.0)
            if ts.shape[0] > 0:
                dt_input = np.asarray(ts[-1] - ts[0])
            xs, ys, ts, ps = self.event_formatting(xs, ys, ts, ps)

            # data augmentation
            xs, ys, ps = self.augment_events(xs, ys, ps, batch)

            # events to tensors
            event_cnt = self.create_cnt_encoding(xs, ys, ps)
            event_mask = self.create_mask_encoding(xs, ys, ps)
            event_voxel = self.create_voxel_encoding(xs, ys, ts, ps)
            event_list = self.create_list_encoding(xs, ys, ts, ps)
            event_list_pol_mask = self.create_polarity_mask(ps)

            # hot pixel removal
            if self.config["hot_filter"]["enabled"]:
                hot_mask = self.create_hot_mask(event_cnt, batch)
                hot_mask_voxel = torch.stack([hot_mask] * self.num_bins, axis=2).permute(2, 0, 1)
                hot_mask_cnt = torch.stack([hot_mask] * 2, axis=2).permute(2, 0, 1)
                event_voxel = event_voxel * hot_mask_voxel
                event_cnt = event_cnt * hot_mask_cnt
                event_mask *= hot_mask.view((1, hot_mask.shape[0], hot_mask.shape[1]))

            # load frames when required
            if self.config["data"]["mode"] == "frames":
                curr_idx = int(np.floor(self.batch_row[batch]))
                next_idx = int(np.ceil(self.batch_row[batch] + self.config["data"]["window"]))

                frames = np.zeros((2, self.config["loader"]["resolution"][0], self.config["loader"]["resolution"][1]))
                img0 = self.open_files[batch]["images"][self.open_files_frames[batch].names[curr_idx]][:]
                img1 = self.open_files[batch]["images"][self.open_files_frames[batch].names[next_idx]][:]
                frames[0, :, :] = self.augment_frames(img0, batch)
                frames[1, :, :] = self.augment_frames(img1, batch)
                frames = torch.from_numpy(frames.astype(np.uint8))

            # load GT optical flow when required
            dt_gt = 0.0
            if self.config["data"]["mode"] == "gtflow_dt1" or self.config["data"]["mode"] == "gtflow_dt4":
                idx = int(np.ceil(self.batch_row[batch] + self.config["data"]["window"]))
                if self.config["data"]["mode"] == "gtflow_dt1":
                    flowmap = self.open_files[batch]["flow_dt1"][self.open_files_flowmaps[batch].names[idx]][:]
                elif self.config["data"]["mode"] == "gtflow_dt4":
                    flowmap = self.open_files[batch]["flow_dt4"][self.open_files_flowmaps[batch].names[idx]][:]
                flowmap = self.augment_flowmap(flowmap, batch)
                flowmap = torch.from_numpy(flowmap.copy())
                if idx > 0:
                    dt_gt = self.open_files_flowmaps[batch].ts[idx] - self.open_files_flowmaps[batch].ts[idx - 1]
            dt_gt = np.asarray(dt_gt)

            # update window
            self.batch_row[batch] += self.config["data"]["window"]

            # break while loop if everything went well
            break

        # prepare output
        output = {}
        output["event_cnt"] = event_cnt
        output["event_voxel"] = event_voxel
        output["event_mask"] = event_mask
        output["event_list"] = event_list
        output["event_list_pol_mask"] = event_list_pol_mask
        if self.config["data"]["mode"] == "frames":
            output["frames"] = frames
        if self.config["data"]["mode"] == "gtflow_dt1" or self.config["data"]["mode"] == "gtflow_dt4":
            output["gtflow"] = flowmap
        output["dt_gt"] = torch.from_numpy(dt_gt)
        output["dt_input"] = torch.from_numpy(dt_input)

        output["gt_time"] = gt_time  # laurens
        output["gt_dt"] = gt_dt  # laurens
        output["gt_translation"] = t  # laurens
        output["gt_rotation"] = r  # laurens

        return output
