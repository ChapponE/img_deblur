import textwrap
import re

import numpy
from deepinv.utils import cal_psnr
from torch.utils.data import DataLoader

from src.utils.data_utils import otfA_, import_problem
from src.utils.data_utils import import_data_problem


import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import json
import random
import scipy


def find_available_folder(save_path):
    folder_count = 0
    while True:
        folder_name = f"train_{folder_count}"
        folder_path = os.path.join(save_path, folder_name)
        if not os.path.exists(folder_path):
            return folder_name
        folder_count += 1

def available_UnfoldedFB_folder(fix=True, notfix=True, dataset=[0.5, 7, 0.25]):
    # all Unfolded models trained in Unfolded folder in results
    dataset_params = f"sb={dataset[0]}_s={dataset[1]}_sn={dataset[2]}"
    save_path = r"C:\Users\levovo pro p50\Documents\informatique\IA\img_deblur\data\results\UnfoldedFB"
    folder_list_tmp = os.listdir(save_path)
    folder_list = []

    if fix:
        folder_list = [folder for folder in folder_list_tmp if ('fix' in folder and dataset_params in folder)]
    if notfix:
        folder_list += [folder for folder in folder_list_tmp if ('fix' not in folder and dataset_params in folder)]
    folder_count = [len(os.listdir(os.path.join(save_path, folder))) for folder in folder_list]
    folder_zip = zip(folder_list, folder_count)
    dictionary = {f'{folder}': [i for i in range(count)] for folder, count in folder_zip}
    print(dictionary)
    return dictionary

def UnfoldedFB_init(depth, scales, dataset=[0.5, 7, 0.25], denoiser_name='UNet', device_='cpu'):
    from src.models.unfolded_FB.model import UnfoldedFB

    base_dir = '../../data'

    otfA = otfA_(*dataset, device_=device_)

    model = UnfoldedFB(otfA, depth=depth, denoiser_name=denoiser_name,
                       scales=scales)

    return (model)

def UnfoldedFBfix_init(depth, scales, dataset=[0.5, 7, 0.25], denoiser_name='UNet', device_='cpu'):
    from src.models.unfolded_FB.model_fix import UnfoldedFB

    base_dir = '../datas'

    otfA = otfA_(*dataset, device=device_)

    model = UnfoldedFB(otfA, depth=depth, denoiser_name=denoiser_name,
                       scales=scales)
    return (model)

def extract_param_model(path, add_path=None):
    json_filepath = os.path.join(path, 'model_config.json')

    if add_path is not None:
        json_filepath = os.path.normpath(os.path.join(add_path, json_filepath))

    with open(json_filepath, 'r') as f:
        params = json.load(f)
    param_model = params['param_model']

    return param_model


def extract_model_path(folder, n_train, add_path=None):
    base_dir = '../../data'

    if add_path is not None:
        base_dir = os.path.join(add_path, base_dir)

    if folder.split('_')[0] in ['UnfoldedFB', 'UnfoldedFBfix']:
        results_dir = os.path.join(base_dir, 'results', 'UnfoldedFB')

    folder_modeltype = os.path.join(results_dir, folder)

    if folder.split('_')[0] in ['UnfoldedFB', 'UnfoldedFBfix']:
        folder_modeltype_train = os.path.join(folder_modeltype, f'train_{n_train}')

    return os.path.normpath(folder_modeltype_train)

def load_weights(folder, n_train, device_='cpu'):

    folder_modeltype_train = extract_model_path(folder, n_train)

    param_model = extract_param_model(folder_modeltype_train)

    if param_model['model_name'] == 'UnfoldedFBfix':
        model = UnfoldedFBfix_init(param_model['depth'], param_model['additional_params']['scales'], param_model['module_name'],   device_=device_)
    elif param_model['model_name'] == 'UnfoldedFB':
        model = UnfoldedFB_init(param_model['depth'], param_model['additional_params']['scales'], extract_problem_variable(folder), device_=device_)
    checkpoint_path = os.path.join(folder_modeltype_train, f"ckp_{param_model['epoch']-1}.pth.tar")
    checkpoint = torch.load(checkpoint_path, map_location=model.device)
    model.load_state_dict(checkpoint['state_dict'])

    return model

def modificate_model_name(model_name, modeltype=False):
    if modeltype:
        if model_name.split('_')[0] == 'UnfoldedFB':
            return 'FB'+'notfix' + '_' + model_name.split('_')[5] + '_' + model_name.split('_')[6]
        if model_name.split('_')[0] == 'UnfoldedFBfix':
            return 'FB'+'fix' + '_' + model_name.split('_')[5] + '_' + model_name.split('_')[6]
    else:
        if model_name.split('_')[0] == 'UnfoldedFB':
            return 'notfix' + '_' + model_name.split('_')[5]  + '_' + model_name.split('_')[6]
        if model_name.split('_')[0] == 'UnfoldedFBfix':
            return 'fix' + '_' + model_name.split('_')[5] + '_' + model_name.split('_')[6]

def import_model(model_name, n_train):
    if model_name.split('_')[0] in ['UnfoldedFB', 'UnfoldedFBfix']:
        model = load_weights(model_name, n_train)
    return model

def reformat(estimated_img, detached=False, squeeze=False, unsqueeze=False, swap_dim=False, torchtonumpy=False, numpytotorch=False, tocpu=False, nb_unsqueeze=1, return_numpy=False ):
        if numpytotorch:
            estimated_img = torch.from_numpy(estimated_img)
        if  type(estimated_img)==torch.Tensor:
            if detached:
                estimated_img = estimated_img.detach()
            if squeeze:
                estimated_img = estimated_img.squeeze(0)

            if swap_dim:
                estimated_img = estimated_img.transpose(len(estimated_img.shape)-1, 0)
            if unsqueeze:
                for i in range(nb_unsqueeze):
                    estimated_img = estimated_img.unsqueeze(0)
            if tocpu:
                estimated_img = estimated_img.detach().to('cpu')

            if torchtonumpy:
                    estimated_img = estimated_img.detach().to('cpu').numpy()

        return estimated_img

def import_history(path, epoch):
    checkpoint_path = os.path.join(path, f"ckp_{epoch-1}.pth.tar")
    checkpoint = torch.load(checkpoint_path)

    loss_history = checkpoint['loss']
    psnr_history = checkpoint['psnr']
    return loss_history, psnr_history

def plot_psnr_loss(ax1, loss_history, psnr_history, title, axes_label=True, ylim=None, y2lim=None):
        try:
            if y2lim is None:
                y2lim = [psnr_history.min(), psnr_history.max()]
            if ylim is None:
                ylim = [min(loss_history), max(loss_history)]
        except:
            if y2lim is None:
                y2lim = [min(psnr_history), max(psnr_history)]
            if ylim is None:
                ylim = [min(loss_history), max(loss_history)]
        import textwrap
        ax1.plot([j for j in range(len(loss_history))], loss_history, '-', label='loss')
        ax1.set_ylim(ylim)
        ax1.set_yscale('log')
        ax1.set_title('\n'.join(textwrap.wrap(title, width=20)), fontsize=25, loc='center', fontweight='bold')

        ax1_bis = ax1.twinx()
        ax1_bis.plot([j for j in range(len(psnr_history))], psnr_history, 'r-', label='psnr')
        #label
        if axes_label:
            ax1_bis.set_ylabel('Value of psnr', fontsize=20, fontweight='bold')
            ax1.set_xlabel('Number of epochs', fontsize=20, fontweight='bold')
            ax1.set_ylabel('Value of MSE', fontsize=20, fontweight='bold')
        # Add legends for both subplots
        ax1.legend(loc='upper left')
        ax1_bis.legend(loc='upper right')
        ax1_bis.set_ylim(y2lim)

def plot_loss(ax1, loss_history, title):
    import textwrap

    ax1.plot([j for j in range(len(loss_history))], loss_history, '-')
    ax1.set_ylim(1e-6, 100)
    ax1.set_yscale('log')
    ax1.set_title('\n'.join(textwrap.wrap(title, width=30)), fontsize=20, loc='center')
    ax1.set_xlabel('Number of epochs')
    ax1.set_ylabel('Value of MSE')

def calculate_psnr(model, dataset_test, physics):
    dataloader = DataLoader(dataset=dataset_test, batch_size=1)
    psnr_list = []
    for xs, ys in dataloader:
        xs = xs.to(model.device)
        ys = ys.to(model.device)
        xs_recon = reformat(model(ys, physics), unsqueeze=True)
        ys = reformat(ys.to(model.device), unsqueeze=True)
        psnr_list += [cal_psnr(xs, xs_recon)]
    return np.mean(psnr_list)


def extract_problem_variable(model_name):
    # Define a regular expression pattern to match 'sb', 's', and 'sn' values
    pattern = r'sb=([\d.]+)_s=([\d.]+)_sn=([\d.]+)'

    # Use re.search to find the pattern in the model name
    match = re.search(pattern, model_name)

    if match:
        sb = float(match.group(1))
        s = int(match.group(2))
        sn = float(match.group(3))
    return [sb, s, sn]

def test_models(models_dict, sample=False, plot=True, force_to_save=False, nb_underscore=None, ploting_squared=False, plot_CV_PnP_table=False, ploting_row=False, set_psnr_ax=None, reverse_table=False):
    ''' models_dict is a dictionary of the form: {model1:[n_train1, n_train2, ...], model2:[n_train1, n_train2, ...], ...}
        sample is a sample of the dataset to test the models on. If False, a random sample is taken.
        plot is a boolean to plot the results or not
        force_to_save is a boolean to force the saving of the loss and psnr history imagrd and all inside all_data.mat'''

    with torch.no_grad():
        device_ = 'cpu'
        # define figures
        n_max_init = 0
        n_tot_init = 0

        for model_name, train_list in sorted(models_dict.items()):
            n_max_init = min(max(n_max_init, len(train_list)),3)
            n_tot_init += min(len(train_list), 3)
        n_models = len(models_dict)

            # Create a figure for loss and psnr
        fig1 = plt.figure(figsize=(5*(n_max_init), 5*n_models))

            # Create a figure for a reconstruction sample
        fig3 = plt.figure(figsize=((n_tot_init+2)*5, 5))

        table_psnr = []
        n_params = []
        model_names = []
        training_times = []
        max_psnr = -np.inf
        model_counter = 0
        #Loop over models
        # models_dict_sorted = personalized_sort(models_dict)
        for i, (model_name, train_list) in enumerate(sorted(models_dict.items())):
            table_psnr += [[]]

            training_times += [[]]
            if nb_underscore is not None:
                model_names += [modificate_model_name(model_name)]
            else:
                model_names += [model_name]
            # Loop over inits
            for j, n_train in enumerate(train_list):
                if n_train>2:
                    continue
                print(model_name, n_train)

                # Import params of the model:
                model_path = extract_model_path(model_name, n_train)
                try:
                    param_model = extract_param_model(model_path)
                    dataset_params = extract_problem_variable(model_name)
                    param_problem = import_data_problem(*dataset_params)

                except:
                    print('problem with param')
                    continue
                # if all_datas.mat doesn't exists in model_path, create it
                try:
                    thing_to_save = scipy.io.loadmat(os.path.join(model_path, 'all_datas.mat'))

                except Exception as e:
                    thing_to_save = {}

                #Import model
                model = import_model(model_name, n_train)

                try:
                    training_times[i].append([round(param_model['time']/3600, 2)])
                except:
                    pass

                if i == 0 and j == 0:
                    # Import problem
                    param_problem = import_data_problem(*dataset_params)
                    dataset_test, physics, nm_img = import_problem(*dataset_params, train=False, n_img=True)
                    if sample != False:
                        img =  sample
                    else:
                        idx = random.randint(0, len(dataset_test) - 1)
                        img = dataset_test[idx]
                    original_img, degraded_img = img[0].to(device_), img[1].to(device_)

                    # Plot original sample
                    if plot:
                        for w in range(1):
                            ax3 = fig3.add_subplot(1, n_tot_init+2, w*(n_tot_init+2) + 1)
                            ax3.imshow(original_img[w].to('cpu'), cmap='gray', vmin=0, vmax=1)
                            ax3.set_title(textwrap.fill('Original sample', 30)+ f'\n ', fontsize=25, fontweight='bold')
                            ax3.axis('off')

                        # Plot degraded sample
                        for w in range(1):
                            ax3 = fig3.add_subplot(1, n_tot_init+2, w*(n_tot_init+2) + 2)
                            ax3.imshow(degraded_img[w].to('cpu'), cmap='gray', vmin=0, vmax=1)
                            ax3.set_title(textwrap.fill(f'Degraded sample', 30)+ f'\n PSNR: {round(cal_psnr(reformat(original_img[w], unsqueeze=True, nb_unsqueeze=2), reformat(degraded_img[w], unsqueeze=True, nb_unsqueeze=2)), 2)}', fontsize=25, fontweight='bold')
                            ax3.axis('off')

                # Import psnr and loss history
                if param_model['trainable']:
                    loss_history, psnr_history = import_history(model_path, param_model['epoch'])
                    if not 'loss' in thing_to_save or force_to_save:
                        thing_to_save.update({'loss': loss_history})
                    if psnr_history is not None and (not 'psnr_history' in thing_to_save or force_to_save):
                        thing_to_save.update({'psnr_history': psnr_history})

                # Plot loss and psnr history
                if plot and not ploting_squared and not ploting_row:
                    ax1 = fig1.add_subplot(n_models, n_max_init, i*n_max_init + j + 1)
                    model_name = param_model['full_model_name']

                    if param_model['trainable']:
                        try:
                            if j == 0:
                                axes_label = True
                            else:
                                axes_label = False
                            plot_psnr_loss(ax1, loss_history, psnr_history, model_names[i], axes_label=axes_label)
                        except:
                            plot_loss(ax1, loss_history, model_names[i])

                elif ploting_row:
                    if model_counter==0:
                        fig4 = plt.figure(figsize=(5*4 , 5))
                    order_plot = model_counter+1
                    ax4 = fig4.add_subplot(1, 4, order_plot)
                    model_name = param_model['full_model_name']
                    modificated_model_name = model_name.split('_')[0] + '_' + modificate_model_name(model_name)
                    if order_plot == 1:
                        axes_label = True
                    else:
                        axes_label = False

                    if param_model['trainable']:

                        try:
                            plot_psnr_loss(ax4, loss_history, psnr_history, modificated_model_name, axes_label=axes_label, y2lim=set_psnr_ax)
                        except:
                            plot_loss(ax4, loss_history, modificated_model_name)

                # Calculate psnr if it is not in the json
                try:
                    psnr = round(param_model['psnr'], 2)

                except:
                    print('psnr wasn t saved')
                    psnr = round(calculate_psnr(model, dataset_test, physics), 2)
                    param_model['psnr'] = psnr
                    #edit the json:
                    json_filepath = os.path.join(model_path, 'model_config.json')
                    with open(json_filepath, 'r') as f:
                        params = json.load(f)
                    params['param_model'] = param_model
                    with open(json_filepath, 'w') as f:
                        json.dump(params, f, indent=True)
                if not 'psnr' in thing_to_save or force_to_save:
                    thing_to_save.update({'psnr': psnr})

                table_psnr[i].append(str(psnr))

                # Select best model
                if psnr > max_psnr:
                    max_psnr = psnr
                    best_model_name = model_name
                    best_model_n_train = n_train

                # Plot reconstruction sample
                if plot:
                    estimated_img = model(degraded_img.unsqueeze(0), physics)

                    for w in range(1):
                        psnr_ = round(cal_psnr(reformat(original_img[w, ...], detached=True, unsqueeze=True, nb_unsqueeze=2,
                                                tocpu=True),
                                       reformat(estimated_img[0, w, ...], detached=True, unsqueeze=True,
                                                nb_unsqueeze=2, tocpu=True)), 2)
                        ax3 = fig3.add_subplot(1, n_tot_init + 2, w * (n_tot_init + 2) + model_counter + 3)
                        ax3.axis('off')
                        ax3.imshow(reformat(estimated_img[0, w, ...], detached=True, squeeze=False, swap_dim=False,
                                            tocpu=True), cmap='gray', vmin=0, vmax=1)
                        if w == 0:
                            ax3.set_title(textwrap.fill(f'{modificated_model_name}', 20) + f'\n PSNR: {psnr_}', fontsize=25, va='top', fontweight='bold')
                        else:
                            ax3.set_title(textwrap.fill(f'PSNR: {round(cal_psnr(reformat(original_img[w, ...], detached=True, unsqueeze=True, nb_unsqueeze=2, tocpu=True), reformat(estimated_img[0, w, ...], detached=True, unsqueeze=True, nb_unsqueeze=2, tocpu=True)),2)}', 30), fontsize=25, fontweight='bold')

                model_counter += 1

                # Save the number of parameters
                try:
                    if type(thing_to_save['n_param']) == numpy.ndarray:
                        thing_to_save['n_param'] = thing_to_save['n_param'][0][0]
                    if not 'n_param' in thing_to_save or force_to_save:
                        thing_to_save.update({'n_param': param_model['n_parameters']})

                except:
                    if not 'n_param' in thing_to_save or force_to_save:
                        thing_to_save.update({'n_param': 0})
                try:
                    if not 'training_time' in thing_to_save or force_to_save:
                        thing_to_save.update({'training_time': param_model['time']})
                except:
                    pass
                # Save the updated data back to all_datas.mat
                scipy.io.savemat(os.path.join(model_path, 'all_datas.mat'), thing_to_save)

            try:
                # add empty table cells
                if param_model['trainable']:
                    n_params.append(param_model['n_parameters'])
                elif 'PnPPGD' in model_name:
                    n_params.append(thing_to_save['n_param'])
                else:
                    n_params.append(0)
                ############## Coment the while anf loop with for ##############
                # for k in range(n_max_init - min(len(train_list), 3)):
                while len(table_psnr[i]) < n_max_init:
                    table_psnr[i].append('')

                training_times[i] = np.round(np.mean(training_times[i]),2)

            except:
                training_times[i] = 'None'

        if plot:
            # tableau psnr
            fig2 = plt.figure(figsize=(15, n_models*2))
            ax2 = fig2.add_subplot(111)

            # Création du tableau

            train_numbers = list(np.arange(n_max_init)) + ['mean'] + ['std'] + ['n_params'] + ['training_time (h)']

            #add mean and std of each list in table_psnr
            table_psnr_mean = list(np.zeros_like(table_psnr))
            for i, model_psnr in enumerate(table_psnr):
                try:
                    table_psnr_mean[i] = model_psnr + [round(np.mean([float(psnr_tmp) for psnr_tmp in model_psnr if psnr_tmp != '']), 2)] + [round(np.std([float(psnr_tmp) for psnr_tmp in model_psnr if psnr_tmp != '']), 2)] + [n_params[i]] + [training_times[i]]
                except:
                    table_psnr_mean[i] = (len(train_numbers))*['']
            if plot_CV_PnP_table:
                train_numbers =  ['mean']  + ['n_params'] + [
                    'training_time (h)']
                table_psnr_mean = list(np.zeros_like(table_psnr))
                for i, model_psnr in enumerate(table_psnr):
                    table_psnr_mean[i] = [
                        round(np.mean([float(psnr_tmp) for psnr_tmp in model_psnr if psnr_tmp != '']), 2)]  + [
                                             n_params[i]] + [training_times[i]]

            modificated_model_names = [modificate_model_name(model_name, modeltype=True) for model_name in model_names]

            # table = ax2.table(cellText=table_psnr_mean,
            #                  rowLabels=modificated_model_names,
            #                  colLabels=train_numbers,
            #                  loc='center')
            # Reverse the data and labels
            if reverse_table:
                table_psnr_mean.reverse()
                modificated_model_names.reverse()

            # Create the table
            fig, ax2 = plt.subplots()
            table = ax2.table(
                cellText=table_psnr_mean,
                rowLabels=modificated_model_names,
                colLabels=train_numbers,
                loc='center'
            )

            # make green best model and red worst model:
            for i, model_psnr in enumerate(table_psnr):
                if len(model_psnr) > 1:
                    for j, psnr in enumerate(model_psnr):
                        if psnr == max(table_psnr[i]):
                            cell_max = table[i+1, j]
                        elif psnr == min(table_psnr[i]):
                            cell_min = table[i+1, j]
                    # Set the color of the digit
                    try:
                        cell_min.set_text_props(fontweight='bold', color='red')
                    except:
                        pass
                    cell_max.set_text_props(fontweight='bold', color='green')

            # Personnalisation du tableau
            table_ax = table.auto_set_column_width(range(len(train_numbers)))
            table.auto_set_font_size(True)
            table.set_fontsize(7)
            table.scale(1, 1)
            table.axes.axis(False)

            # Affichage des figures
            fig1.tight_layout()
            fig2.tight_layout()
            fig3.tight_layout()
            if ploting_squared or ploting_row:
                fig4.tight_layout()
            plt.show()
    try:
        return {f'{best_model_name}': [best_model_n_train]}
    except:
        pass
