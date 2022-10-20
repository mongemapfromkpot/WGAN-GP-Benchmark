import torch
from torch import autograd
from scipy import stats
import numpy as np

 

class Grad_Pen():
    
    def __init__(self, gp_lambda, device):
        assert gp_lambda > 0
        self.lamb = gp_lambda
        self.device = device
        self.sampler = stats.uniform.rvs
    
    def __call__(self, critic, generated, real):
        batch = real.shape[0]
        
        # Get points where gradient will be penalized
        t = torch.from_numpy(self.sampler(size=(batch,1,1,1)).astype(np.float32)).to(self.device)
        gp_points = (1-t)*generated + t*real
        gp_points.requires_grad = True
        
        # Compute gradients
        critvals = critic(gp_points)
        gradients = autograd.grad(outputs = critvals, 
                                  inputs = gp_points,
                                  grad_outputs = torch.ones_like(critvals),
                                  create_graph = True)
        gradients = gradients[0].view(gradients[0].size(0), -1)
        grad_pen = (torch.clamp(gradients.norm(2, dim=1) - 1, min=0)**2).mean() * self.lamb
        return grad_pen