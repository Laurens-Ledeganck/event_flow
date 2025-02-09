# TODO: 
# Task A: clean up code, set up clear pipeline
# Task B: convert more rotation data 
# Task C: investigate MSE discrepancy 
# Task D: train EVFLowNet
# Task E: try different losses, architectures to obtain good performance

import argparse
import datetime

import mlflow
import torch
from torch.optim import *
import numpy as np

from configs.parser import YAMLParser
from event_flow_pipeline.dataloader.h5 import H5Loader
from event_flow_pipeline.loss.flow import EventWarping
from event_flow_pipeline.models.model import (
    FireNet,
    RNNFireNet,
    LeakyFireNet,
    FireFlowNet,
    LeakyFireFlowNet,
    E2VID,
    EVFlowNet,
    RecEVFlowNet,
    LeakyRecEVFlowNet,
    RNNRecEVFlowNet,

    LIFFireNet,
    PLIFFireNet,
    ALIFFireNet,
    XLIFFireNet,
    LIFFireFlowNet,
    SpikingRecEVFlowNet,
    PLIFRecEVFlowNet,
    ALIFRecEVFlowNet,
    XLIFRecEVFlowNet,
)
from event_flow_pipeline.utils.gradients import get_grads
from event_flow_pipeline.utils.utils import load_model, save_csv, save_diff, save_model, rename_run, log_to_overview
from event_flow_pipeline.utils.visualization import Visualization

from new_files.laurens_stuff import (
    FullRotationModel,
    RotationLoss, 
    ModifiedH5Loader
)


def train(args, config_parser):
    mlflow.set_tracking_uri(args.path_mlflow)

    # configs
    config = config_parser.config
    if config["data"]["mode"] == "frames":
        print("Config error: Training pipeline not compatible with frames mode.")
        raise AttributeError

    # log config
    mlflow.set_experiment(config["experiment"])
    mlflow.start_run()
    mlflow.log_params(config)
    mlflow.log_param("prev_runid", args.prev_runid)
    config = config_parser.combine_entries(config)

    run_id = mlflow.active_run().info.run_id
    run_name = mlflow.active_run().info.run_name
    artifact_uri = mlflow.active_run().info.artifact_uri
    mlflow.end_run()
    rename_run(path=artifact_uri[:artifact_uri.index(run_id)], run_id=run_id, run_name=run_name)
    mlflow.start_run(run_id=run_name)

    print("MLflow dir:", mlflow.active_run().info.artifact_uri[:-9])

    # log git diff
    save_diff("train_diff.txt")

    # initialize settings
    device = config_parser.device
    kwargs = config_parser.loader_kwargs

    # visualization tool
    if config["vis"]["enabled"]:
        vis = Visualization(config)

    # model initialization and settings
    rotation = False
    model_args = config["model"].copy()
    if "prev_runid" in config["model"].keys() and config["model"]["use_existing"]:
        rotation = True # special case if we want to tap out rotation
        try: 
            prev_model = load_model(config["model"]["prev_runid"], None, device)
            prev_model.eval()
        except AttributeError:
            raise ValueError("prev_runid not found")

        model_args["flow_model"] = prev_model
        model_args["batch_size"] = config["loader"]["batch_size"]
        model_args["resolution"] = config["loader"]["resolution"]
        model_args["rotation_mode"] = config["loader"]["rotation_mode"]
        model_args["rotation_type"] = config["loader"]["rotation_type"]

    model = eval(model_args["name"])(model_args).to(device)
    model = load_model(args.prev_runid, model, device)

    model.train()

    # optimizers
    optimizer = eval(config["optimizer"]["name"])(model.parameters(), lr=config["optimizer"]["lr"])
    optimizer.zero_grad()

    # loss function
    if rotation: 
        loss_function = RotationLoss(config, device)
    else: 
        loss_function = EventWarping(config, device)
    
    # data loader
    if rotation: 
        data = ModifiedH5Loader(config, config["model"]["num_bins"], config["model"]["round_encoding"])
    else: 
        data = H5Loader(config, config["model"]["num_bins"], config["model"]["round_encoding"])
    dataloader = torch.utils.data.DataLoader(
        data,
        drop_last=True,
        batch_size=config["loader"]["batch_size"],
        collate_fn=data.custom_collate,
        worker_init_fn=config_parser.worker_init_fn,
        **kwargs,
    )

    # create first log
    log_to_overview({
        'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'run_name': run_name, 
        'model_type': config['model']['name'],
        'architecture': str(model).replace('\n', ' '),
        'params': config
    }, path=args.path_mlflow)

    # simulation variables
    train_loss = 0
    best_loss = 1.0e6
    end_train = False
    grads_w = []

    # training loop
    data.shuffle()
    while True:
        for inputs in dataloader:

            if data.new_seq:
                data.new_seq = False

                loss_function.reset()
                model.reset_states()
                optimizer.zero_grad()

            if data.seq_num >= len(data.files):
                mlflow.log_metric("loss", train_loss / (data.samples + 1), step=data.epoch)

                with torch.no_grad():
                    if train_loss / (data.samples + 1) < best_loss:
                        save_model(model)
                        best_loss = train_loss / (data.samples + 1)
                        log_to_overview({
                            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'run_name': run_name, 
                            'model_type': config['model']['name'],
                            'loss': best_loss,
                            'architecture': str(model).replace('\n', ' '),
                            'params': config
                        }, path=args.path_mlflow, replace_last=True)

                data.epoch += 1
                data.samples = 0
                train_loss = 0
                data.seq_num = data.seq_num % len(data.files)

                # save grads to file
                if config["vis"]["store_grads"]:
                    save_csv(grads_w, "grads_w.csv")
                    grads_w = []

                # finish training loop
                if data.epoch == config["loader"]["n_epochs"]:
                    end_train = True

            # forward pass
            x = model(inputs["event_voxel"].to(device), inputs["event_cnt"].to(device))

            # event flow association
            if rotation: 
                loss_function.prepare_loss(
                    x["rotation"],
                    inputs["gt_rotation"].to(device)
                )

            else: 
                loss_function.event_flow_association(
                    x["flow"],
                    inputs["event_list"].to(device),
                    inputs["event_list_pol_mask"].to(device),
                    inputs["event_mask"].to(device),
                )

            # backward pass
            if loss_function.num_events >= config["data"]["window_loss"] or rotation:

                # overwrite intermediate flow estimates with the final ones
                if config["loss"]["overwrite_intermediate"]:
                    loss_function.overwrite_intermediate_flow(x["flow"])

                # loss
                loss = loss_function()
                if np.isnan(loss.item()): 
                    raise ValueError(f"Loss became NaN; y_true was {inputs['gt_rotation']}, y_pred was {x['rotation']}")
                train_loss += loss.item()

                # update number of loss samples seen by the network
                data.samples += config["loader"]["batch_size"]

                loss.backward()

                # clip and save grads
                if config["loss"]["clip_grad"] is not None:
                    torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), config["loss"]["clip_grad"])
                if config["vis"]["store_grads"]:
                    grads_w.append(get_grads(model.named_parameters()))

                optimizer.step()
                optimizer.zero_grad()

                # mask flow for visualization
                flow_vis = x["flow"][-1].clone()
                if model.mask and config["vis"]["enabled"] and config["loader"]["batch_size"] == 1:
                    flow_vis *= loss_function.event_mask

                model.detach_states()
                loss_function.reset()

                # visualize
                with torch.no_grad():
                    if config["vis"]["enabled"] and config["loader"]["batch_size"] == 1:
                        vis.update(inputs, flow_vis, None)

            # print training info
            if config["vis"]["verbose"]:
                print(
                    "Train Epoch: {:04d} [{:03d}/{:03d} ({:03d}%)] Loss: {:.6f}".format(
                        data.epoch,
                        data.seq_num,
                        len(data.files),
                        int(100 * data.seq_num / len(data.files)),
                        train_loss / (data.samples + 1),
                    ),
                    end="\r",
                )

        if end_train:
            break

    mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/train_flow.yml",
        help="training configuration",
    )
    parser.add_argument(
        "--path_mlflow",
        default="results/mlruns",
        help="location of the mlflow ui",
    )
    parser.add_argument(
        "--prev_runid",
        default="",
        help="pre-trained model to use as starting point",
    )
    args = parser.parse_args()

    # launch training
    train(args, YAMLParser(args.config))
