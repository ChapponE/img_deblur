
from src.utils.data_utils import import_data_problem
from src.utils.train import find_available_folder
from src.utils.data_utils import import_problem
from src.utils.data_utils import otfA_
from src.models.unfolded_FB.model import UnfoldedFB
from src.utils.train import train

import time
import torch
from torch.utils.data import DataLoader
import deepinv as dinv
from pathlib import Path
import json
from datetime import date

device = 'cpu'
# dataset
sb = 0.5
s = 7
sn = 0.25
dataset_params = [sb, s, sn]
param_problem = import_data_problem(*dataset_params)

#Settings model
list_depth = [5]
list_scales = [2]

for depth in list_depth:
    for scales in list_scales:
        print(f"depth={depth}, scales={scales}")

        param_model = {}
        param_model['model_name'] = 'UnfoldedFB'
        param_model['module_name'] = 'UNet'
        param_model['depth'] = depth
        param_model['trainable'] = True
        param_model['date'] = str(date.today())
        param_model['additional_params'] = {'scales': scales}
        additional_params_string = '_'.join(
            [f"{key}={round(value, 2) if isinstance(value, float) else value}" for key, value in
             param_model['additional_params'].items()])
        param_model['full_model_name'] = f"{param_model['model_name']}_{param_model['module_name']}_{param_problem['name_dataset']}_depth={param_model['depth']}_{additional_params_string}"

        # Others:
        batch_size = 2
        param_model['epoch'] = 50
        param_model['lr'] = 0.0001  # learning rate
        losses = [dinv.loss.SupLoss(metric=dinv.metric.mse())]  # loss
        optimizer= torch.optim.Adam  # optimiz
        scheduler = False  # scheduler
        wandb_vis = False

        # Directories
        BASE_DIR = '../../data'
        result_model_dir = f"{BASE_DIR}/results/UnfoldedFB/{param_model['full_model_name']}"
        n_train = find_available_folder(result_model_dir)
        RESULTS_MODEL_DIR = Path(f"{result_model_dir}/{n_train}")

        # Import problem
        n_data_train = 600
        n_data_test = 50

        dataset_train, physics = import_problem(*dataset_params, choose_device=device)
        dataset_test, physics = import_problem(*dataset_params, train=False, choose_device=device)
        dataset_train = torch.utils.data.Subset(dataset_train, range(n_data_train))
        dataset_test = torch.utils.data.Subset(dataset_test, range(n_data_test))
        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

        # model
        otfA = otfA_(*dataset_params, device)
        model = UnfoldedFB(otfA, depth=param_model['depth'], denoiser_name=param_model['module_name'],
                           **param_model['additional_params'])
        optimizer = optimizer(model.parameters(), lr=param_model['lr'])
        param_model['n_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Timer
        tic = time.time()
        # train the network
        train(
            model=model,
            train_dataloader=dataloader_train,
            epochs=param_model['epoch'],
            losses=losses,
            eval_dataloader=dataloader_test,
            physics=physics,
            optimizer=optimizer,
            device=device,
            ckp_interval=int(param_model['epoch']),
            save_path=RESULTS_MODEL_DIR,
            plot_images=False,
            verbose=True,
            wandb_vis=wandb_vis,
        )

        toc = time.time()

        # save time
        param_model['time'] = toc - tic

        # Save information on the model:
        with open(RESULTS_MODEL_DIR / "model_config.json", 'w') as file:
            json.dump({'param_model': param_model, 'param_problem': param_problem}, file, indent=True)
        with open(RESULTS_MODEL_DIR / 'model_config.json', 'r') as file:
            params = json.load(file)
