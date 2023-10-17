import json
import torch
import os
import deepinv as dinv

def import_data_problem(sb=0.5, size=7, sn=0.25):
    name_dataset = f'sb={sb}_s={size}_sn={sn}'
    dir_relatif = f'../../data/datasets_dinv/{name_dataset}'
    dir_absolue = os.path.join(os.path.dirname(__file__), dir_relatif)
    dir_absolue_norm = os.path.normpath(dir_absolue)
    path_problem = str(os.path.join(dir_absolue_norm, 'param_problem.json')).replace('\\', '/')
    with open(path_problem, 'r') as fp:
        param_problem = json.load(fp)
    return param_problem

def psf2otf(psf, output_shape=None):
    psfsize = torch.tensor(psf.shape)
    if output_shape is not None:
        outsize = torch.tensor(output_shape)
        padsize = outsize - psfsize
    else:
        padsize = psfsize.new_zeros(psfsize.shape)
        outsize = psfsize

    psf = torch.nn.functional.pad(psf, (0, padsize[1], 0, padsize[0]), 'constant')

    for i in range(len(psfsize)):
        psf = torch.roll(psf, -int(psfsize[i] / 2), i)
    otf = torch.fft.fftn(psf)
    nElem = torch.prod(psfsize).item()
    nOps = 0
    for k in range(len(psfsize)):
        nffts = nElem / psfsize[k].item()
        nOps = nOps + psfsize[k].item() * torch.log2(psfsize[k].clone().detach()) * nffts
    if torch.max(torch.abs(torch.imag(otf))) / torch.max(torch.abs(otf)) <= nOps * torch.finfo(torch.float32).eps:
        otf = torch.real(otf)
    return otf

def otfA_(sb=0.5, size=7, sn=0.25, device_='cpu'):
    param_problem = import_data_problem(sb, size, sn)
    kernel = torch.Tensor(param_problem['kernel']).to(device_)
    otfA = psf2otf(kernel[0,0,...], param_problem['shape'])
    return otfA

# Import problem = model, physics
def import_problem(sb=0.5, s=7, sn=0.25 , train=True, choose_device=False, n_img=False):
    # Directory
        #dataset
    chemin_dossier_absolue = os.path.dirname(__file__)
    dir_dinv_dataset_relatif = f'../../data/datasets_dinv/sb={sb}_s={s}_sn={sn}/dinv_dataset0.h5'
    dir_dinv_dataset_absolue = os.path.join(chemin_dossier_absolue, dir_dinv_dataset_relatif)
    dir_dinv_dataset_absolue_norm = os.path.normpath(dir_dinv_dataset_absolue)

        #nm_img
    dir_nm_img_relatif = f'../../data/datasets_dinv/sb={sb}_s={s}_sn={sn}/param_problem.json'
    dir_nm_img_absolue = os.path.join(chemin_dossier_absolue, dir_nm_img_relatif)
    dir_nm_img_absolue_norm = os.path.normpath(dir_nm_img_absolue)

    #dataset_train:
    dataset = dinv.datasets.HDF5Dataset(path=dir_dinv_dataset_absolue_norm, train=train)
    # Device
    if choose_device == False:
        device_ = 'cpu'
    else:
        device_ = choose_device

    # Import physics:
    param_problem = import_data_problem(sb, s, sn)
    blur = torch.tensor(param_problem['kernel']).to(device_)
    physics = dinv.physics.BlurFFT(dataset[0][0].shape, blur, device=device_)
    physics.noise_model = dinv.physics.GaussianNoise(sigma=sn)

    # Import nm_img:
    with open(dir_nm_img_absolue_norm, 'r') as file:
        params = json.load(file)
    if n_img:
        nm_img = params['nm_img']
        return dataset, physics, nm_img
    else:
        return dataset, physics