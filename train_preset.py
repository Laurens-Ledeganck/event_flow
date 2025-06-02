"""
Wrapper for train.py, can be used to schedule multiple runs with slightly different parameters. 
Configure a base config in configs.base_config.yml with all your parameters, then specify in preset.csv the values of the parameters for each run. 
An empty value means 
"""
#TODO: add file description

from types import SimpleNamespace
import pandas as pd
import numpy as np
import yaml
import os
import mlflow

from configs.parser import YAMLParser
from train import train


def load_settings(csv_path):
    preset_configs = pd.read_csv(csv_path, sep=',')

    start_idx = preset_configs.index[preset_configs['logic'].str.upper()=='START'].tolist()[0]
    stop_idx = preset_configs.index[preset_configs['logic'].str.upper()=='STOP'].tolist()[0]
    preset_configs = preset_configs.iloc[start_idx:stop_idx+1].reset_index(drop=True)
    return preset_configs


def config_processing(original, new):

    if new is None or new == "" or (isinstance(new, float) and np.isnan(new)):
        new = original

    elif str(new).lower() == "nan" or str(new).lower() == "none" or str(new).lower() == "null" or str(new).lower() == "/" or str(new).lower() == "\\": 
        new = None

    else: 
        if original is None: 
            data_type = str
        else: 
            data_type = type(original)
            if isinstance(original, list):
                new = str(new).strip(' ').split(',')
                for i in range(len(new)):
                    new[i] = config_processing(original[i] if i < len(original) else original[-1], new[i])
        new = data_type(new)

    return new


def modify_config(settings, base_config: dict):

    new_config = {}
    new_config.update(base_config)
    for key in base_config.keys():

        if isinstance(base_config[key], dict):
            for subkey in base_config[key].keys():
                if hasattr(settings, key+'.'+subkey):
                    new_config[key][subkey] = config_processing(base_config[key][subkey], settings[key+'.'+subkey])
                    #if new_config[key][subkey] is None: new_config[key].pop(subkey)  # remove NaN values
        else: 
            if hasattr(settings, key):
                new_config[key] = config_processing(base_config[key], settings[key])
                #if new_config[key] is None: new_config.pop(key)  # remove NaN values

    return new_config


if __name__ == "__main__":
    # load preset parameters
    base_config_file = 'configs/train_flow.yml' #'configs/base_config.yml'
    base_config = YAMLParser(base_config_file).config
    prev_config = base_config
    preset_configs = load_settings('preset.csv') 

    # start loop
    logs = "Good news! \nYour ML runs were completed successfully. Find the logs below:"
    for run_nr in range(len(preset_configs)):
        settings = preset_configs.iloc[run_nr]
        stop = True if str(settings["logic"]).upper() == "STOP" else False

        try:
            # Loading config:
            new_config = modify_config(settings, prev_config)
            prev_config = new_config
            with open('configs/temp_config.yml', 'w') as file:
                yaml.dump(new_config, file)
                
            # Initialize training:
            print(f"\nStarting run {run_nr+1} out of {len(preset_configs)}\n")
            args = SimpleNamespace(
                config='configs/temp_config.yml', 
                path_mlflow='results/mlruns', 
                prev_runid='',  # TODO, but optional
                use_wandb='True',
                note= None if not hasattr(settings, 'note') else settings.note
            )
            config_parser = YAMLParser(args.config) 

            # Run training loop:
            run_log = train(args, config_parser, alert=["Runs completed!", logs] if stop else None) 
            print(f"Run {run_nr+1} done.")


        except Exception as e:
           run_log = f"Run failed due to {e}"
           mlflow.end_run()

        logs += "\n"    
        logs += run_log

    # End loop

    if os.path.isfile('configs/temp_config.yml'):
        os.remove('configs/temp_config.yml')

    print("\nRuns completed.")
    print(logs)


# ----- FOR TESTING ------ #
"""
if __name__ == "__main__":
    base_config_file = 'configs/train_flow.yml'
    #base_config = YAMLParser(base_config_file).config
    args = SimpleNamespace(
        config=base_config_file, 
        path_mlflow='results/mlruns', 
        prev_runid='',  # TODO, but optional
        use_wandb='True',
        note= None
    )
    config_parser = YAMLParser(args.config) 
    train(args, config_parser, alert=None)
"""


"""
if __name__ == "__main__":
    while True: 
        base_config_file = 'configs/train_flow.yml'
        base_config = YAMLParser(base_config_file).config
        settings = load_settings('preset.csv').iloc[0]
        new_config = modify_config(settings, base_config)
        with open('configs/temp_config.yml', 'w') as file:
            yaml.dump(new_config, file)
        print(type(base_config['spiking_neuron']), type(new_config['spiking_neuron']))
        input("Press Enter to try again ")
"""
