import torch
from torchvision import utils
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')



def generate_image(im_name, data, ext, path):
    """
    After re-centering data, this code saves a grid of samples to the specified path.
    Saves grid under path/im_name.ext.
    """

    data = 0.5*data + 0.5*torch.ones_like(data) # by default, data is generated in [-1,1]

    grid = utils.make_grid(data, nrow = 16, padding = 1)
    plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))
    plt.axis('off')
    plt.savefig(os.path.join(path,'{}.{}'.format(im_name, ext)), dpi = 300)
    plt.close()



def save_individuals(b_idx, data, ext, path, to_rgb=False):
    """
    After re-centering data, this code individually saves numbered images in the specified path.
    Optionally will repeat greyscale images 3 times along colour channel
    """
    data = 0.5*data + 0.5*torch.ones_like(data) # by default, data is generated in [-1,1]
    bs = data.shape[0]
   
    for i in range(bs):
        single = data[i,:,:,:]
        if single.shape[0] == 1 and to_rgb:
            single = single.repeat(3,1,1)
        utils.save_image(single, os.path.join(path, '{:05d}.{}'.format(b_idx*bs + i, ext)))