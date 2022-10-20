"""
Convenience functions that return new network with given architectures. 
"""


###############
### INFOGAN ###
###############
# Generator
def infogan_g(**kwargs):
    num_chan = kwargs['num_chan']
    if 'im_res' in kwargs:
        h = w = kwargs['im_res']
    elif ('h' in kwargs) and ('w' in kwargs):
        h = kwargs['h']
        w = kwargs['w']
    else:
        raise ValueError("Image resolution must be specified when initializing InfoGAN generator, either through 'im_res' (for square image) or through 'h' and 'w'.")
    
    from GAN_utils.architectures.infogan import Generator
    generator = Generator(num_chan, h, w)
    generator.latent_dim = 64
    return generator

# Critic
def infogan_c(**kwargs):
    num_chan = kwargs['num_chan']
    if 'im_res' in kwargs:
        h = w = kwargs['im_res']
    elif ('h' in kwargs) and ('w' in kwargs):
        h = kwargs['h']
        w = kwargs['w']
    else:
        raise ValueError("Image resolution must be specified when initializing InfoGAN generator, either through 'im_res' (for square image) or through 'h' and 'w'.")
    try: 
        dim = kwargs['dim']
    except KeyError:
        dim = 64
    
    from GAN_utils.architectures.infogan import Discriminator
    critic = Discriminator(num_chan, h, w, dim)
    return critic





###############
### SNDCGAN ###
###############
# Generator
def sndcgan_g(**kwargs):
    num_chan = kwargs['num_chan']
    if 'im_res' in kwargs:
        h = w = kwargs['im_res']
    elif ('h' in kwargs) and ('w' in kwargs):
        h = kwargs['h']
        w = kwargs['w']
    else:
        raise ValueError("Image resolution must be specified when initializing InfoGAN generator, either through 'im_res' (for square image) or through 'h' and 'w'.")
    try: 
        dim = kwargs['dim']
    except KeyError:
        dim = 128
    
    from GAN_utils.architectures.sndcgan import Generator
    generator = Generator(num_chan, h, w, input_dim=dim)
    generator.latent_dim = dim
    return generator
    

# Critic
def sndcgan_c(**kwargs):
    num_chan = kwargs['num_chan']
    if 'im_res' in kwargs:
        h = w = kwargs['im_res']
    elif ('h' in kwargs) and ('w' in kwargs):
        h = kwargs['h']
        w = kwargs['w']
    else:
        raise ValueError("Image resolution must be specified when initializing InfoGAN generator, either through 'im_res' (for square image) or through 'h' and 'w'.")
    try: 
        dim = kwargs['dim']
    except KeyError:
        dim = 64
    
    from GAN_utils.architectures.sndcgan import Discriminator
    critic = Discriminator(num_chan, h, w, dim)
    return critic

