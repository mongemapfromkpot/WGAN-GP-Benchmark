"""
Main GAN training function for GAN training.
"""

from GAN_utils import training_stats
import torch



def train_iters(networks,
                optimizers,
                grad_penalty,
                latent_dim,
                loader,
                loader_gen,
                iters,
                critters,
                skip_last=False):
    """
    Will train WGAN until the generator has been updated the number of times specified by 'iters' (minus one if skip_last==True).
    Before each update of the generator, the critic is trained for a number of iterations specified by 'critters'. 
    """
    # ------------------------------- Setting up -------------------------------
    generator = networks[0]
    critic = networks[1]
    optim_g = optimizers[0]
    optim_c = optimizers[1]
   
    use_cuda = next(generator.parameters()).is_cuda
    
    # ------------------------------- Training -------------------------------
    for iteration in range(iters):
        #####################
        # (1) Update critic #
        #####################
        # Reset all parameters of all critics to requires_grad=True. (They are set to false when updating the generators)
        for p in critic.parameters():  
            p.requires_grad = True 
        
        for i in range(critters):
            # Reset critic gradients
            for param in critic.parameters(): 
                param.grad = None   # more efficient than zero_grad()
            
            # Get minibatches of real and fake data. 
            try:
                real = loader_gen.next()[0]
            except StopIteration:
                loader_gen = iter(loader)
                real = loader_gen.next()[0]
            noise = torch.randn(loader.bs, latent_dim)
            if use_cuda:
                real = real.cuda()
                noise = noise.cuda()
            fake = generator(noise).detach() # We detach to avoid useless backpropagation through generator
            
            # Compute and backpropagate critic values on real and fake data
            critval_r = critic(real).mean()
            critval_r.backward()    
            critval_f = -critic(fake).mean()
            critval_f.backward()  
            
            # Compute and backpropagate gradient penalty
            grad_pen = grad_penalty(critic, real, fake)
            grad_pen.backward()
            
            # Optimizer step
            optim_c.step()
    
            if i == (critters-1):
                critic_div = critval_r - critval_f
        
        ########################
        # (2) Update generator #
        ########################    
        # Turn off grad computations for critic
        for p in critic.parameters():
            p.requires_grad = False  # to avoid computation
        
        # Reset generator gradient
        for param in generator.parameters():
            param.grad = None # more efficient than netG.zero_grad()
        
        # Get minibatch of noise inputs and generate fake data
        noise = torch.randn(loader.bs, latent_dim)
        if use_cuda:
            noise = noise.cuda()
        fake = generator(noise)
        
        # Compute and backpropagate critic value on fake data
        critval = critic(fake).mean()
        critval.backward()
        
        # Optimizer step
        optim_g.step()
        
        # Keep training statistics
        training_stats.plot('Critic_div', critic_div.cpu().data.numpy())
        training_stats.plot('Gradient_penalty', grad_pen.cpu().data.numpy())
        training_stats.plot('Critic_fakes', critval.cpu().data.numpy())
        training_stats.tick()