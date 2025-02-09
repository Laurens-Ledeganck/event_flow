# Adapted to work on both Windows & Linux
# And to include functions 
#  - to replace run_id with run_name (for easier directory navigation) 
#  - to log some desired metrics to a csv file 

import os
from functools import partial  # change
import csv  # change

import mlflow
import pandas as pd
import torch


def load_model(prev_runid, model, device):
    try:
        run = mlflow.get_run(prev_runid)
    except:
        print("Failed to get run ", prev_runid)  # change
        return model

    artifact_uri = os.path.join(run.info.artifact_uri, 'model', 'data', 'model.pth')  # change
    if os.name == 'nt':  # check if we're running on Windows  # change
        artifact_uri = artifact_uri.replace("/", "\\")  # change
    model_dir = os.path.join(os.getcwd(), artifact_uri)  # change
    #model_dir = os.path.join(os.getcwd(), artifact_uri[artifact_uri.find('mlruns'):])

    if model_dir[:7] == "file://":
        model_dir = model_dir[7:]

    print("Loading model from dir: ", model_dir)  # change

    if os.path.isfile(model_dir):
        model_loaded = torch.load(model_dir, map_location=device) 
        if model:  # change
            model.load_state_dict(model_loaded.state_dict()) 
        else:  # change
            model = model_loaded  # change
        print("Model restored from " + prev_runid + "\n")
    else:
        print("No model found at " + prev_runid + "\n")

    return model


def create_model_dir(path_results, runid):
    path_results = os.path.join(path_results, runid)  # change
    if not os.path.exists(path_results):
        os.makedirs(path_results)
    print("Results stored at " + path_results + "\n")
    return path_results


def rename_run(path, run_id, run_name):  # new function
    if run_id in os.listdir(path):

        os.rename(os.path.join(path, run_id), os.path.join(path, run_name))

        with open(os.path.join(path, run_name, 'meta.yaml'), 'r') as file:
            lines = file.readlines()
        
        lines[0] = lines[0].replace(run_id, run_name)
        lines[5] = lines[5].replace(run_id, run_name)

        with open(os.path.join(path, run_name, 'meta.yaml'), 'w') as file:
            file.writelines(lines)


def save_model(model):
    mlflow.pytorch.log_model(model, "model")
    print("\nsaved model \n")  # change


def save_csv(data, fname):
    # create file if not there
    path = mlflow.get_artifact_uri(artifact_path=fname)
    if path[:7] == "file://":  # to_csv() doesn't work with 'file://'
        path = path[7:]
    if not os.path.isfile(path):
        mlflow.log_text("", fname)
        pd.DataFrame(data).to_csv(path)
    # else append
    else:
        pd.DataFrame(data).to_csv(path, mode="a", header=False)


def save_diff(fname="git_diff.txt"):
    # .txt to allow showing in mlflow
    path = mlflow.get_artifact_uri(artifact_path=fname)
    if path[:7] == "file://":
        path = path[7:]
    mlflow.log_text("", fname)
    os.system(f"git diff > {path}")


def log_to_overview(params, replace_last=False, path='results/mlruns'):  # new function
    with open(os.path.join(path, 'overview_runs.csv'), 'r') as file:
        reader = csv.reader(file, delimiter=';')
        rows = list(reader)
        heading = rows[0]
    
    row = []
    for param in heading:
        row += [params[param] if param in params.keys() else None]
    if replace_last:
        rows = rows[:-1] + [row]
    else: 
        rows = rows + [row]
        
    with open(os.path.join(path, 'overview_runs.csv'), 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';')
        writer.writerows(rows)
