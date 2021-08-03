import json
import os
import sys
import time
import torch
import torch.backends.cudnn as cudnn

from training.training import Trainer


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Get config file from command line arguments
if len(sys.argv) != 2:
    raise(RuntimeError("Wrong arguments, use python main.py <config_path>"))
config_path = sys.argv[1]

# Open config file
with open(config_path) as f:
    config = json.load(f)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    if config["gpu"] == 0:
        device = 'cuda:0'
    if config["gpu"] == 1:
        device = 'cuda:1'
    else:
        device = 'cuda'
else:
    device = 'cpu'


if config["path_to_data"] == "":
    raise(RuntimeError("Path to data not specified. Modify path_to_data attribute in config to point to data."))

if config["train"] > 0:
    # Create a folder to store experiment results
    timestamp = time.strftime("%Y-%m-%d_%H-%M")
    directory = "{}_{}".format(timestamp, config["id"])
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Save config file in experiment directory
    with open(directory + '/config.json', 'w') as f:
        json.dump(config, f)

else:
    directory = config["test"]["test_path"]


cudnn.benchmark = True

if config["dataset"] == "AFHQ":
    path_to_data = config["path_to_data"]
    resolution = config["resolution"]
    training = config["training"]
    test = config["test"]

    if config["train"]:
        train = True
        batch_size = training["batch_size"]
    else:
        train = False
        batch_size = test["batch_size"]
else:
    raise(RuntimeError("Requested Dataset unfound"))


trainer = Trainer(device, 
                train= train,
                directory = directory,
                path_to_data = path_to_data,
                batch_size = batch_size,
                size = resolution,
                max_iters = training["max_iters"],
                resume_iters = training["resume_iters"],
                restored_model_path = training["restored_model_path"],
                lr = training["lr"],
                weight_decay = training["weight_decay"],
                beta1 = training["beta1"],
                beta2 = training["beta2"],
                milestones = training["milestones"],
                scheduler_gamma = training["scheduler_gamma"],
                rec_weight = training["rec_weight"],
                X_kld_weight = training["X_kld_weight"],
                S_kld_weight = training["S_kld_weight"],
                inter_kld_weight = training["inter_kld_weight"],
                print_freq = training["print_freq"],
                sample_freq = training["sample_freq"],
                model_save_freq = training["model_save_freq"],
                test_iters = test["test_iters"],
                test_path = test["test_path"])

if train:
    trainer.train()
else:
    trainer.test()
    


