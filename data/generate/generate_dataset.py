import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import torch
import matplotlib.pyplot as plt
import deepinv as dinv
import json
import os

device = "cpu"

# Class Dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_array):
        super().__init__()
        #self.data_array = torch.from_numpy(data_array).to(dinv.device)
        # add channel dim
        self.data_array = np.expand_dims(data_array, axis=1)
    def __getitem__(self, index):
        x = torch.from_numpy(self.data_array[index,:,:,:]).type(torch.float).to(device)
        return x
    def __len__(self):
        return self.data_array[:,0,0,0].shape[0]

# import mnist and transform in tensor
(x_train, y_train), (x_test, y_test) = mnist.load_data()
shape = x_train[0].shape
x_train = Dataset(x_train/255)
x_test = Dataset(x_test/255)
print(x_train[0].shape)
# Define physics
#----------------------
size = 7
sigma_blur = 0.5
sigma_noise = 0.25
#----------------------
def gaussian_kernel(size, sigma):
    """Generate a 2D Gaussian kernel."""
    kernel = torch.Tensor([[np.exp(-((x - size // 2) ** 2 + (y - size // 2) ** 2) / (2 * sigma ** 2)) for x in range(size)] for y in range(size)])
    return kernel.unsqueeze(0).unsqueeze(0) / torch.sum(kernel)
kernel = gaussian_kernel(size, sigma_blur)
physics = dinv.physics.BlurFFT( (1, *shape), kernel)
physics.noise_model = dinv.physics.GaussianNoise(sigma=sigma_noise)

# Save blur kernel and information about dataset in json
name_dataset = f'sb={sigma_blur}_s={size}_sn={sigma_noise}'
dir = f'../datasets_dinv/{name_dataset}'
os.makedirs(dir, exist_ok=True)

param_problem = {
    'kernel': kernel.numpy().tolist(),
    'shape': shape,
    'sigma_blur': sigma_blur,
    'sigma_noise': sigma_noise,
    'size_blur': size,
    'n_data_train': len(x_train),
    'n_data_test': len(x_test),
    'n_channels': 1,
    'name_dataset': name_dataset
}
json_file_path = os.path.join(dir, "param_problem.json")
with open(json_file_path, 'w') as fp:
    json.dump(param_problem, fp, indent=4)

# Save dataset
num_workers = 4 if torch.cuda.is_available() else 0  # set to 0 if using small cpu, else 4
dinv.datasets.generate_dataset(train_dataset=x_train, test_dataset=x_test,
   physics=physics, device=device, save_dir=dir, num_workers=num_workers, supervised=True)

# Load dataset and param_problem
data = dinv.datasets.HDF5Dataset(path=f'{dir}/dinv_dataset0.h5', train=True)
with open(f'{dir}/param_problem.json', 'r') as fp:
    param_problem = json.load(fp)
print(param_problem['kernel'])

# plot
img = data[0]
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img[0][0,...], cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(img[1][0,...], cmap='gray')
plt.show()




