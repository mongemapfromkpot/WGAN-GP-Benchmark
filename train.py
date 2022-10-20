"""
Train InfoGAN or SNDCGAN
"""

import os
import shutil
import time
import argparse
import logging
import pandas as pd
import random
import numpy as np
import torch

from pytorch_fid import inception
from GAN_utils import dataloader, networks, gradient_penalty, training, evaluating, training_stats, generate_samples



def folder_tool(path, required_files):
    """
    Checks if experiment folder already exists, and if so, what's in it.
    Returns:
        False:   If folder does not exists.
        True:    If folder does exist and contains required files. 
        Error:   If path_name exists as a file, or as a folder that doesn't contain required files  
    """
    if not os.path.exists(path):
        os.makedirs(path)
        return False # Will start from scratch
    
    if not os.path.isdir(path):
        raise OSError("Given folder path already exists but is not a directory. ABORT")
    
    if not all(file in os.listdir(path) for file in required_files):
        raise OSError("Given folder already exists, but does not contain all necessarry files to resume prior training")
    
    return True # Will resume previous training

    


def main(args, device):    
    
    # ------------------------------- SETTING UP ------------------------------------------------
    # Check if folder exists, and if so whether resuming is possible.
    folder = args.folder
    
    required_files = [f'generator_{args.generator}_last.pth', f'critic_{args.critic}_last.pth', 'metrics.pkl']
    resume = folder_tool(folder, required_files)
    
    img_folder = os.path.join(folder, 'generated_images')
    os.makedirs(img_folder, exist_ok=True)
    if args.temp_folder==None:
        temp_folder = os.path.join(folder, 'temp')
        os.makedirs(temp_folder, exist_ok=True)
    else:
        temp_folder = args.temp_folder
    iterations = args.iterations
    
    # Logging
    log_file = os.path.join(folder, 'log.txt')
    
    logging.basicConfig(level = 0)
    logger = logging.getLogger(__name__) # We use a single logger with 2 handlers.
    logger.propagate = False
    logger.setLevel(0)
    
    console_handler = logging.StreamHandler() # Writes to console
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    console_handler.setLevel(1)
    logger.addHandler(console_handler)
    
    file_handler = logging.FileHandler(log_file) # Writes to log.txt
    file_handler.setFormatter(logging.Formatter('%(message)s'))
    file_handler.setLevel(2)
    logger.addHandler(file_handler)
    
    if resume:
        metrics_frame = pd.read_pickle(os.path.join(folder, 'metrics.pkl'))
        start_iter = list(metrics_frame['FID'].keys())[-1]
        last_iter = start_iter + iterations
        training_stats.load(folder)
        training_stats.offset(start_iter)
        message = f"RESUMING FROM ITERATION {start_iter}\nWill train up to iteration {last_iter}\n"
        message += f"Dataset: {args.dataset}\n"
        message += "State dictionaries reloaded from generator_last.pth and critic_last.pth\n"
    else:
        metrics_frame = pd.DataFrame()
        metrics_frame.insert(0, "FID", []) 
        start_iter = 0
        last_iter = iterations
        message = "TRAINING FROM SCRATCH\n"
        message += f"Dataset: {args.dataset}\nGenerator model: {args.generator}\nCritic model: {args.critic}\n"
    message += f"Device: {device}"
    logger.log(2, message)
    
    # Checkpoints are iterations where metrics will be computed.
    num_checkpoints = args.checkpoints
    checkpoints = []
    for i in range(num_checkpoints):
        checkpoints.append((i+1) * iterations//num_checkpoints + start_iter)
    logger.log(2, f"Checkpoints where FID will be evaluated (in generator iterations):\n{checkpoints} \n\n...\n")
    
    # Deterministic behavior
    seed = args.seed
    if seed != -1: #if non-default seed
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark=False # If true, optimizes convolution for hardware, but gives non-deterministic behaviour
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    
    
    
    # ------------------------------- DEFINITIONS ------------------------------------------------
    logger.log(1, "Starting definitions")
  
    # Loader
    logger.log(1, "loader...")
    loader = getattr(dataloader, args.dataset)(args, train=True)
    loader_gen = iter(loader)
    
    # Networks
    logger.log(1, "networks...")
    params_g = {}
    params_g['im_res'] = loader.im_res
    params_g['num_chan'] = loader.channels
    model_g = args.generator + '_g'
    generator = getattr(networks, model_g)(**params_g).to(device)
    
    params_c = {}
    params_c['im_res'] = loader.im_res
    params_c['num_chan'] = loader.channels
    model_c = args.critic + '_c'
    critic = getattr(networks, model_c)(**params_c).to(device)
    
    if resume:
        logger.log(1, "Loading network state dicts from last time...")
        generator.load_state_dict(torch.load(os.path.join(folder, f'generator_{args.generator}_last.pth')))
        critic.load_state_dict(torch.load(os.path.join(folder, f'critic_{args.critic}_last.pth')))
    nets = (generator, critic)
    
    # Optimizers
    logger.log(1, "optimizers...")
    optim_g = torch.optim.Adam(generator.parameters(), lr=args.glr, betas=(0.5, 0.999))
    optim_c = torch.optim.Adam(critic.parameters(),    lr=args.dlr, betas=(0.5, 0.999))
    
    if resume:
        if 'optim_g_last.pth' in os.listdir(folder):
            optim_g.load_state_dict(torch.load(os.path.join(folder, 'optim_g_last.pth')))
        if 'optim_c_last.pth' in os.listdir(folder):
            optim_c.load_state_dict(torch.load(os.path.join(folder, 'optim_c_last.pth')))
    optimizers = (optim_g, optim_c)
    
    # Gradient penalty
    logger.log(1, "gradient penalty...")
    grad_penalty = gradient_penalty.Grad_Pen(gp_lambda=args.gp_lambda, device=device)
    
    # For FID evaluations
    if resume and 'real_data_stats.npz' in os.listdir(folder):
        real_data = os.path.join(folder, 'real_data_stats.npz')
    else:
        real_data = args.test_data_path
    inception_net = inception.InceptionV3([inception.InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device)

    # Noise constants
    logger.log(1, "noise constants...")
    fid_noise = torch.randn(10000, generator.latent_dim, device=device)   # Used for generated data when evaluating FID 
    img_noise = torch.randn(64, generator.latent_dim, device=device)      # Used to generate sample images
    
    
    
    # ------------------------------- MAIN EVENT ------------------------------------------------
    logger.log(1, "Starting training...\n")
    starttime = time.perf_counter()
    total_train_time = 0
    current_iter = start_iter
    best_fid = 99999999 # Hopefully we won't do worse than that...
    best_iter = 0
    
    generator.train()
    critic.train()
    
    for i in range(len(checkpoints)):
        
        ### TRAIN
        logger.log(1, f'Now training from generator iteration {current_iter} to {checkpoints[i]}...')
        torch.cuda.reset_peak_memory_stats(device=device)
        
        time_ = time.perf_counter()
        num_iters = checkpoints[i]-current_iter 
        training.train_iters(networks=nets,
                             optimizers=optimizers,
                             grad_penalty=grad_penalty,
                             latent_dim=generator.latent_dim,
                             loader=loader,
                             loader_gen=loader_gen,
                             iters=num_iters,
                             critters=args.critters)
        time_ = time.perf_counter() - time_
        total_train_time += time_
        time_per_iter = time_/num_iters
        max_mem_gb = torch.cuda.max_memory_allocated(device='cuda')/1073741824
        message = f"Trained from iteration {current_iter} to {checkpoints[i]} in {round(time_)} seconds.\n"
        message += f"({time_per_iter:.2f} sec/iter)\n"
        message += f"Max GPU memory usage: {max_mem_gb:.2f} gb.\n"
        logger.log(2, message)
        current_iter += num_iters
        training_stats.flush(folder)
        
        
        ### TEST
        logger.log(1, f'Evaluating FID after generator iteration {current_iter}')
        time_ = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device=device)
        
        with torch.no_grad():
        
            # Generate image
            logger.log(1, "Generating images")
            gen_imgs = generator(img_noise).detach().cpu()
            generate_samples.generate_image(f'iter_{current_iter}', gen_imgs, 'jpg', img_folder)
            
            # Compute FID
            logger.log(1, "Evaluating FID")
            fid, real_data = evaluating.fid_eval(generator=generator,
                                                 real_data=real_data,
                                                 directory=temp_folder,
                                                 inception_net=inception_net,
                                                 noise=fid_noise)
            message = f"FID after iteration {current_iter}: {fid:.2f}"
            if fid < best_fid:
                message += f' ...best yet! \nSaving generator and critic model dicts under generator_{args.generator}_best.pth and critic_{args.critic}_best.pth.'
                best_fid = fid
                best_iter = current_iter
                train_time_to_best = total_train_time
                torch.save(generator.state_dict(), os.path.join(folder, f'generator_{args.generator}_best.pth'))
                torch.save(critic.state_dict(), os.path.join(folder, f'critic_{args.critic}_best.pth'))
        
        series = pd.Series(name=current_iter, dtype=float)
        series['FID'] = fid
        metrics_frame = metrics_frame.append([series])
        
        time_ = time.perf_counter() - time_
        max_mem_gb = torch.cuda.max_memory_allocated(device='cuda')/1073741824
        message += f"\nTime to generate image and evaluate FID: {round(time_)} seconds.\n"
        message += f"Max GPU memory used during checkpoint: {max_mem_gb:.2f} gb.\n\n...\n"
        logger.log(2, message)
    
    
        
    # ------------------------------- EPILOGUE ------------------------------- 
    metrics_frame.to_pickle(os.path.join(folder, 'metrics.pkl'))
    if not 'real_data_stats.npz' in os.listdir(folder):
        shutil.copyfile(os.path.join(temp_folder, 'real_data_stats.npz'),
                        os.path.join(folder, 'real_data_stats.npz'))
    if best_iter == current_iter:
        os.rename(os.path.join(folder, f'generator_{args.generator}_best.pth'),
                  os.path.join(folder, f'generator_{args.generator}_last.pth'))
        os.rename(os.path.join(folder, f'critic_{args.critic}_best.pth'),
                  os.path.join(folder, f'critic_{args.critic}_last.pth'))
        message = "All done. Best FID obtained at the last evaluation, so saving only last state dictionaries\n"
    
    else:
        torch.save(generator.state_dict(), os.path.join(folder, f'generator_{args.generator}_last.pth'))
        torch.save(critic.state_dict(), os.path.join(folder, f'critic_{args.generator}_last.pth'))
        message = "All done. Best FID not obtained at the last evaluation, so saving both best and last state dictionaries\n"
    torch.save(optim_g.state_dict(), os.path.join(folder, 'optim_g_last.pth'))
    torch.save(optim_c.state_dict(), os.path.join(folder, 'optim_c_last.pth'))
    
    total_time = time.perf_counter() - starttime
    message += f"Best FID: {best_fid:.2f}, obtained after {best_iter} iterations, {round(train_time_to_best)} seconds of training (not counting FID checkpoints).\n"
    message += f"Total time required: {round(total_time)} seconds.\n\n\n----------------------\n\n\n"
    logger.log(2, message)
    
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser('InfoGAN and DCGAN experiments')
    
    # General
    parser.add_argument('--folder',             type=str, required=True, help='Folder to create or from which to resume')
    parser.add_argument('--temp_folder',        type=str, default=None,  help='Temporary folder for computations')
    parser.add_argument('--dataset',            type=str, required=True, help='name of dataset')
    parser.add_argument('--data',               type=str, required=True, help='Directory where training data is located')
    parser.add_argument('--test_data_path',     type=str, required=True, help='Directory where testing data is located')
    parser.add_argument('--iterations',         type=int, required=True, help='Number of generator iterations for which to train')
    parser.add_argument('--checkpoints',        type=int, default=None,  help='Number of times metrics are evaluated')
    parser.add_argument('--num_workers',        type=int, default=1,     help='DataLoader worker processes')
    parser.add_argument('--seed',               type=int, default=-1,    help='Set random seed for reproducibility')
    
    # Networks
    parser.add_argument('--generator',          choices=['infogan','sndcgan'], default='infogan', help='Generator model')
    parser.add_argument('--critic',             choices=['infogan','sndcgan'], default='infogan', help='Generator model')
    
    # Gradient penalty
    parser.add_argument('--gp_lambda',          type=float, default=10., help='WGAN-GP gradient penalty weight')
    
    # Training settings
    parser.add_argument('--glr',                type=float, default=1e-4, help='Generator learning rate')
    parser.add_argument('--dlr',                type=float, default=1e-4, help='Critic learning rate')
    parser.add_argument('--bs',                 type=int, default=64,     help='Batch size')
    parser.add_argument('--critters',           type=int, default=5,      help='Number of critic iterations')
    
    args = parser.parse_args()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    main(args, device)