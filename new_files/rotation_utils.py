"""
Based on Jesse's file
"""
# imports
import math
import torch
import numpy as np
from scipy.spatial.transform import Rotation
from functools import partial

from event_flow_pipeline.models.model_util import copy_states
import event_flow_pipeline.models.spiking_util as spiking
from event_flow_pipeline.models.submodules import ConvGRU, ConvLayer, ConvLayer_, ConvLeaky, ConvLeakyRecurrent, ConvRecurrent
from event_flow_pipeline.models.spiking_submodules import ConvALIF, ConvALIFRecurrent, ConvLIF, ConvLIFRecurrent, ConvPLIF, ConvPLIFRecurrent, ConvXLIF, ConvXLIFRecurrent
from event_flow_pipeline.models.base import BaseModel
from event_flow_pipeline.loss.flow import BaseValidationLoss
from event_flow_pipeline.dataloader.h5 import H5Loader, Frames, ProgressBar, FlowMaps
import h5py


class Linear(torch.nn.Module):
    """Linear layer with activation, similar inputs to LinearLIF."""
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            activation = "ReLU",
            device = None,
            dtype = None, 
            norm = None, 
            BN_momentum = 0.1, 
            w_scale = None
        ):
        super().__init__()
        self.layer = torch.nn.Linear(in_features, out_features, bias, device, dtype)
        if activation is None:
            self.activation = None
        else:
            self.activation = eval('torch.nn.'+activation+'()')

        if w_scale is not None:
            torch.nn.init.uniform_(self.layer.weight, -w_scale, w_scale)
            torch.nn.init.zeros_(self.layer.bias)

        self.norm = norm
        if norm == "BN":
            self.norm_layer = torch.nn.BatchNorm2d(out_features, momentum=BN_momentum)
        elif norm == "IN":
            self.norm_layer = torch.nn.InstanceNorm2d(out_features, track_running_stats=True)

    def forward(self, x, prev_state, residual=0):
                # generate empty prev_state, if None is provided
        if prev_state is None:
            prev_state = torch.tensor(0)  # not used

        out = self.layer(x)

        if self.norm in ["BN", "IN"]:
            out = self.norm_layer(out)

        out += residual
        if self.activation is not None:
            out = self.activation(out)

        return out, prev_state


class LinearLIF(torch.nn.Module):
    # TODO test this
    # Based on the Convolutional spiking LIF cell in spiking_submodules.py
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        activation="arctanspike",
        act_width=10.0,
        leak=(-4.0, 0.1),
        thresh=(0.8, 0.0),
        learn_leak=True,
        learn_thresh=True,
        hard_reset=True,
        detach=True,
        norm=None,
    ):
        super().__init__()

        # shapes
        self.input_neurons = in_features
        self.hidden_neurons = out_features

        # parameters
        self.ff = torch.nn.Linear(self.input_neurons, self.hidden_neurons, bias=bias)
        if learn_leak:
            self.leak = torch.nn.Parameter(torch.randn(self.hidden_neurons, 1, 1) * leak[1] + leak[0])
        else:
            self.register_buffer("leak", torch.randn(self.hidden_neurons, 1, 1) * leak[1] + leak[0])
        if learn_thresh:
            self.thresh = torch.nn.Parameter(torch.randn(self.hidden_neurons, 1, 1) * thresh[1] + thresh[0])
        else:
            self.register_buffer("thresh", torch.randn(self.hidden_neurons, 1, 1) * thresh[1] + thresh[0])

        # weight init
        w_scale = math.sqrt(1 / self.input_neurons)
        torch.nn.init.uniform_(self.ff.weight, -w_scale, w_scale)

        # spiking and reset mechanics
        assert isinstance(
            activation, str
        ), "Spiking neurons need a valid activation, see models/spiking_util.py for choices"
        self.spike_fn = getattr(spiking, activation)
        self.register_buffer("act_width", torch.tensor(act_width))
        self.hard_reset = hard_reset
        self.detach = detach

        # norm
        if norm == "weight":
            self.ff = torch.nn.utils.weight_norm(self.ff)
            self.norm = None
        elif norm == "group":
            groups = min(1, self.input_neurons// 4)  # at least instance norm
            self.norm = torch.nn.GroupNorm(groups, self.input_neurons)
        else:
            self.norm = None

    def forward(self, input_, prev_state, residual=0):
        # input current
        if self.norm is not None:
            input_ = self.norm(input_)
        ff = self.ff(input_)

        # generate empty prev_state, if None is provided
        if prev_state is None:
            prev_state = torch.zeros(2, *ff.shape, dtype=ff.dtype, device=ff.device)
        v, z = prev_state  # unbind op, removes dimension

        # clamp thresh
        thresh = self.thresh.clamp_min(0.01)

        # get leak
        leak = torch.sigmoid(self.leak)

        # detach reset
        if self.detach:
            z = z.detach()

        # voltage update: decay, reset, add
        if self.hard_reset:
            v_out = v * leak * (1 - z) + (1 - leak) * ff
        else:
            v_out = v * leak + (1 - leak) * ff - z * thresh

        # spike
        z_out = self.spike_fn(v_out, thresh, self.act_width)

        return z_out + residual, torch.stack([v_out, z_out])



# first attempt
class SimpleRotationModel(BaseModel):
    # normal ANN
    # TODO: make easier to modify
    # TODO: implement spiking version

    def __init__(self, n_inputs, n_outputs, n_init=0):
        super().__init__()
        self.n_init = n_init
        if self.n_init: n_inputs += self.n_init
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(n_inputs, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, n_outputs)
        )
        
    def forward(self, X, init_rot=None):
        if init_rot: 
            X = torch.concat((X, init_rot))
        return self.layers(X), None
    

# second attempt
class ConvRotationModel(BaseModel):
    # CNN
    # TODO: make easier to modify

    def __init__(self, input_size, n_outputs, n_init=0):
        super().__init__()
        self.n_init = n_init
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(input_size[1], 16, kernel_size=3, stride=2, padding=1),  # 128 -> 64
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            #torch.nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),  # 64 -> 32
            #torch.nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1),  # 32 -> 16
        )

        n_middle = self.conv_layers(torch.randn(input_size)).reshape(input_size[0], -1).shape[1]
        if self.n_init: n_middle += self.n_init

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
    
    def forward(self, X, init_rot=None):
        X = self.conv_layers(X)
        X = X.reshape(X.shape[0], -1)
        if init_rot is not None: 
            X = torch.concat((X, init_rot), dim=-1)
        return self.linear_layers(X), None


class SpikingConvRotationModel(BaseModel):
    """CNN that can be made spiking"""

    def __init__(self, input_size, n_outputs, n_init=0, neurons=[ConvLayer_, Linear], neuron_settings={}):
        # TODO (pick up here): 
        # add prev_states & residuals based on FireNet architecture
        # compare FireNet architecture to EV-FlowNet
        # test without spiking
        # test with spiking
        # investigate hybrid
        # set up new runs
        super().__init__()
        self.n_init = n_init
        self.neuron_settings = neuron_settings  # could contain leak, thresh, learn_leak, learn_thresh and hard_reset

        self.conv1 = neurons[0](input_size[1], 16, kernel_size=3, stride=2, **self.neuron_settings)
        self.conv2 = neurons[0](16, 32, kernel_size=3, stride=2, **self.neuron_settings)
        #self.conv3 = neurons[0](32, 32, kernel_size=3, stride=2, **self.neuron_settings)

        n_middle = self.conv2(
                        self.conv1(torch.randn(input_size), prev_state=None)[0], 
                        prev_state=None)[0].reshape(input_size[0], -1).shape[1]
        if self.n_init: n_middle += self.n_init
        #assert False

        self.linear1 = neurons[1](n_middle, 64)
        self.linear2 = neurons[1](64, 64)
        self.linear3 = neurons[1](64, n_outputs)

        self.layers = torch.nn.Sequential(
            self.conv1, 
            self.conv2, 
            torch.nn.Flatten(),
            self.linear1, 
            self.linear2, 
            self.linear3
        )

        self.n_states = len(self.layers)  # transfer layer will be ignored but should be included so the indices match
        self.reset_states()  # create n_states states set to None
    
    def forward(self, X, init_rot=None, log=False):
        #TODO check if I need residuals -> FireNet doesn't include any
        #TODO check if the reshape/flatten is done correctly: does it handle different batch sizes?
        Xs = [X]
        for i, layer in enumerate(self.layers):
            if isinstance(layer, torch.nn.Flatten):
                Xs += [layer(Xs[-1])]
                #X = X.reshape(X.shape[0], -1)
                if init_rot is not None: 
                    X = torch.concat((X, init_rot), dim=-1)
            else:
                X, self._states[i] = layer(Xs[-1], self._states[i])
                Xs += [X]

        # log activity
        if log:
            # TODO update this when changing the architecture
            activity = {}
            name = [
                "0:input",
                "1:conv1",
                "2:conv2",
                "3:flatten",
                "4:linear1",
                "5:linear2",
                "6:linear3"
            ]
            for n, l in zip(name, [Xs]):
                activity[n] = l.detach().ne(0).float().mean().item()
        else:
            activity = None

        return Xs[-1], activity
    
    @property
    def states(self):
        return copy_states(self._states)

    @states.setter
    def states(self, states):
        self._states = states

    def detach_states(self):
        detached_states = []
        for state in self.states:
            if type(state) is tuple:
                tmp = []
                for hidden in state:
                    tmp.append(hidden.detach())
                detached_states.append(tuple(tmp))
            else:
                detached_states.append(state.detach())
        self.states = detached_states

    def reset_states(self):
        self._states = [None] * self.n_states
    

# integrating everything
class FullRotationModel(BaseModel):
    # the following was adapted from the FireNet code

    def __init__(self, unet_kwargs):
        super().__init__()

        # self.num_bins = unet_kwargs["num_bins"]
        # base_num_channels = unet_kwargs["base_num_channels"]
        # kernel_size = unet_kwargs["kernel_size"]
        # self.encoding = unet_kwargs["encoding"]
        self.norm_input = False if "norm_input" not in unet_kwargs.keys() else unet_kwargs["norm_input"]
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
        self.use_input = unet_kwargs["use_layer_input"]
        self.encoding = unet_kwargs["encoding"]
        self.get_n_transfers()  # will update self.transfer_size and self.n_transfers

        self.include_init = unet_kwargs["include_init"]

        self.rotation_mode = unet_kwargs["rotation_mode"]
        self.rotation_type = unet_kwargs["rotation_type"]
        self.n_outputs = self.get_n_rotation_nodes()

        if unet_kwargs["model_type"] == ('spiking' or 'spiking_conv' or 'conv_spiking' or 'SpikingConvRotationModel'):
            self.model_type = 'spiking'
            kwargs = {}
            if type(unet_kwargs["spiking_neuron"]) is dict:
                kwargs["neurons"] = [ConvLIF, LinearLIF]
                kwargs["neuron_settings"] = unet_kwargs["spiking_neuron"]
            self.rotation_model = SpikingConvRotationModel(input_size=list(self.transfer_size), n_outputs=self.n_outputs, n_init=(self.include_init*self.get_n_rotation_nodes()), **kwargs)
        elif unet_kwargs["model_type"] == ('conv' or 'conv_model' or 'ConvRotationModel'):
            self.model_type = 'conv'
            self.rotation_model = ConvRotationModel(input_size=list(self.transfer_size), n_outputs=self.n_outputs, n_init=(self.include_init*self.get_n_rotation_nodes()))
        else:
            self.model_type = 'linear'
            self.rotation_model = SimpleRotationModel(n_inputs=self.n_transfers, n_outputs=self.n_outputs, n_init=(self.include_init*self.get_n_rotation_nodes()))
    
    def transfer_hook(self, module, inp, output): 
        if self.use_input: 
            values = inp
        else: 
            values = output
        
        if isinstance(values, tuple):
            values = values[0] # TODO investigate where exactly this occurs
        
        if self.transfer_size is None: 
            self.transfer_size = values.shape
        if self.n_transfers is None: 
            self.n_transfers = len(torch.flatten(values[0])) 
        self.current_transfer = values
    
    def register_hook(self, module, transfer_layer:str):
        #print(module, transfer_layer)
        if '.' in transfer_layer:
            idx = transfer_layer.index('.')
            return self.register_hook(getattr(module, transfer_layer[:idx]), transfer_layer[idx+1:])
        else: 
            return getattr(module, transfer_layer).register_forward_hook(self.transfer_hook)
    
    def get_n_transfers(self):
        # first, set the hook on the transfer layer
        hook = self.register_hook(self.flow_model, self.transfer_layer)

        # now pass a dummy input (n_transfers will be registered)
        dummy_voxel = torch.randn(self.input_size, dtype=torch.float32).to(self.device)
        dummy_cnt = torch.randint(0, 3, self.input_size, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outp = self.flow_model(dummy_voxel, dummy_cnt)

        hook.remove()
        return self.n_transfers
    
    def get_n_rotation_nodes(self):
        if self.rotation_type == "rotvec" or self.rotation_type.startswith("euler"):
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

    def forward(self, event_voxel, event_cnt, init_rot=None, log=False):
        """
        :param event_voxel: N x num_bins x H x W
        :param event_cnt: N x 2 x H x W per-polarity event cnt and average timestamp
        :param log: log activity
        :return: output dict with list of [N x 2 X H X W] (x, y) displacement within event_tensor.
        """
        event_cnt = event_cnt.to(self.device)
        event_voxel = event_voxel.to(self.device)   
        if init_rot is not None: init_rot.to(self.device)

        # TODO: potentially log activity
        activity = None

        # forward pass

        # get the hook
        hook = self.register_hook(self.flow_model, self.transfer_layer)
        
        # pass the input through the flow model
        # TODO: cut this off at the transfer layer
        with torch.no_grad():
            flow = self.flow_model(event_voxel, event_cnt, log=log)['flow'][0]

        transfer = self.current_transfer
        if self.norm_input: # normalize input
            mean, stddev = (
                transfer[transfer != 0].mean(),
                transfer[transfer != 0].std(),
            )
            transfer[transfer != 0] = (transfer[transfer != 0] - mean) / stddev
        
        # retrieve the intermediate values to transfer
        if self.model_type == 'linear':
            transfer = torch.reshape(transfer, (transfer.shape[0], self.n_transfers))
        else: 
            transfer = transfer

        # get the output of the rotation model
        if self.include_init: 
            if init_rot is not None: 
                r, activity = self.rotation_model(transfer, init_rot)
            else: 
                raise ValueError("Please provide a valid init_rot")
        else: 
            r, activity = self.rotation_model(transfer)
        

        if not self.use_input:
            hook.remove()

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
            idx1 = np.where(t1 < (file["ground_truth/timestamp"] - file.attrs["t0"]))[0][0]
            
            idx2 = np.where((file["ground_truth/timestamp"] - file.attrs["t0"]) < t2)[0][-1]

            idx1 = idx1 - 1 if idx1 != 0 else idx1
            idx2 = idx2 + 1 if idx2 != len(file["ground_truth/timestamp"])-1 else idx2

            if idx2 - idx1 <= 1: print(f"\nNo ground truth data in file {file} between {t1} and {t2} (closest gt indices are {idx1} and {idx2}). If this occurs often, consider increasing the event window.")

        return idx1, idx2

    def get_time_translation_rotation(self, file, t1, t2):
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
        dtimestamps = timestamps[-1] - timestamps[0]
        timestamp1 = timestamps[0] - file.attrs['gt0']

        t = np.transpose(np.vstack((tx, ty, tz)))
        t1 = t[0]
        self.translation_mode = "difference"  # TODO: add translation_mode property
        if self.translation_mode == "absolute":
            t = t[-1]
            #t = np.mean(t, axis=0)  
        elif self.translation_mode == "difference":
            t = t[-1] - t[0]
        elif self.translation_mode == "zero-offset":
            t = np.mean(t, axis=0) - file.attrs["gt0"][1:4]  # TODO maybe this should also be t[-1]?
        t = torch.flatten(torch.tensor(t, dtype=torch.float32))
        t1 = torch.flatten(torch.tensor(t1, dtype=torch.float32))
        
        # TODO: implement zero-offset properly
        r = np.transpose(np.vstack((qx, qy, qz, qw)))
        r1 = Rotation.from_quat(r[0])
        if self.rotation_mode == "absolute":
            r = Rotation.from_quat(r[-1])
            #r = Rotation.from_quat(np.mean(r, axis=0)) 
        elif self.rotation_mode == "difference":
            r = Rotation.from_quat(r[-1]) * Rotation.from_quat(r[0]).inv()
        elif self.rotation_mode == "local_diff": 
            r = Rotation.from_quat(r[0]).inv() * Rotation.from_quat(r[-1])
        elif self.rotation_mode == "zero-offset":
            r = Rotation.from_quat(np.mean(r, axis=0)) * Rotation.from_quat(file.attrs["gt0"][4:]).inv()  # TODO maybe this should also be r[-1]?
        
        if self.rotation_type == "quat":
            r = r.as_quat()
            r1 = r1.as_quat()
        elif self.rotation_type == "rotvec":
            r = r.as_rotvec()
            r1 = r1.as_rotvec()
        elif self.rotation_type == "matrix":
            r = r.as_matrix()
            r1 = r1.as_matrix()
        elif self.rotation_type == "euler":
            r = r.as_euler("xyz", degrees=False)
            r1 = r1.as_euler("xyz", degrees=False)
        elif self.rotation_type == "euler_deg":
            r = r.as_euler("xyz", degrees=True)
            r1 = r1.as_euler("xyz", degrees=True)
        
        r = torch.flatten(torch.tensor(r, dtype=torch.float32))
        r1 = torch.flatten(torch.tensor(r1, dtype=torch.float32))

        if torch.isnan(t).any() or torch.isnan(r).any():
            raise ValueError(f"NaN value detected in ground truth: t1 = {t1}, t2 = {t2}, idx1 = {idx1}, idx2 = {idx2}, qx = {qx}, t = {r}")

        return timestamp1, dtimestamps, t1, t, r1, r

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
                    gt_time, gt_dt, t_init, t, r_init, r = self.get_time_translation_rotation(self.open_files[batch], ts[0], ts[-1])  # laurens

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

                t = np.empty([0])  # laurens
                t_init = np.empty([0])  # laurens
                r = np.empty([0])  # laurens
                r_init = np.empty([0])  # laurens
                # TODO: check if this is the right approach, the code below seemed more logical but failed
                # if self.rotation_mode == "difference":  # laurens
                #     t = np.empty(len(t))  # laurens
                #     r = np.empty(len(r))  # laurens
                # elif self.rotation_mode == "absolute": # laurens 
                #     t = t_init  # laurens
                #     r = r_init  # laurens

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
        output["gt_t_init"] = t_init  # laurens
        output["gt_translation"] = t  # laurens
        output["gt_r_init"] = r_init  # laurens
        output["gt_rotation"] = r  # laurens

        return output
