import torch
import torch.nn as nn
from deepinv.models import unet
# import src.correct_unet as unet

# Model:
class UnfoldedFB(nn.Module):
    def __init__(self, otfA, depth=20, denoiser_name = 'UNet', **kwargs):
        super(UnfoldedFB, self).__init__()
        self.depth = depth
        self.device = otfA.device
        self.otfA = otfA
        self.tau = torch.ones((self.depth)).to(self.device)
        nom_module = denoiser_name.lower()
        module = globals()[nom_module]
        denoiser = getattr(module, denoiser_name)

        if 'scales' in kwargs:
            self.scales = kwargs['scales']
            # Define the UNet and other layers here
            self.denoiser = nn.ModuleList([denoiser( in_channels=1, out_channels=1, scales=self.scales).to(self.device) for _ in range(self.depth)])

    def forward(self, x, physics):
        # Init
        if physics.__class__.__name__ == 'BlurFFT':
            xk = x
        else:
            xk = physics.A_adjoint(x)

        # Iterate
        for i in range(self.depth):
            # Apply gradient of the datafidelity:
            xk = xk - self.tau[i]*physics.A_adjoint(physics.A(xk)-x)

            # Apply the UNet
            xk = self.denoiser[i](xk)

        return xk


