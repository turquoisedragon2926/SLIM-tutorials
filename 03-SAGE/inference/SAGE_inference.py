import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#from torch_utils import distributed as dist
import dnnlib
from dnnlib.util import tensor_clipping
from training import dataset
from torch_utils.misc import StackedRandomGenerator
import json
from collections import OrderedDict
import warnings
import matplotlib.pyplot as plt
import argparse
import colorcet as cc
import pdb

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

def cdist_masked(x1, x2, mask1=None, mask2=None):
    if mask1 is None or mask2 is None:
        mask1 = torch.ones_like(x1)
        mask2 = torch.ones_like(x2)
    x1 = x1[0].unsqueeze(0)
    diffs = x1.unsqueeze(1) - x2.unsqueeze(0)
    combined_mask = mask1.unsqueeze(1) * mask2.unsqueeze(0)
    error = 0.5 * torch.linalg.norm(combined_mask * diffs)**2
    return error

def get_well_mask(image_shape, num_columns_to_survive, same_for_all_batch=False, device='cuda', seed=None):
    """Creates a mask with random columns being masked.
        Args:
            image_shape: (batch_size, num_channels, height, width)
            num_columns_to_survive: number of columns to be unmasked
            same_for_all_batch: if True, the same mask is applied to all images in the batch
            device: device to use for the mask
            seed: seed for the random number generator
        Returns:
            mask: (batch_size, num_channels, height, width)
    """
    if seed is not None:
        np.random.seed(seed)

    batch_size = image_shape[0]
    num_channels = image_shape[1]
    height = image_shape[2]
    width = image_shape[3]

    # Create an array of zeros with the same width as the image
    corruption_mask = np.zeros(width, dtype=np.float32)

    # Randomly choose columns to survive
    survive_columns = np.random.choice(width, num_columns_to_survive, replace=False)

    # Set the surviving columns to 1
    corruption_mask[survive_columns] = 1

    if same_for_all_batch:
        corruption_mask = corruption_mask.reshape(1, 1, -1)
        corruption_mask = torch.tensor(corruption_mask, device=device, dtype=torch.float32).repeat([batch_size, num_channels, height, 1])
    else:
        corruption_mask = corruption_mask.reshape(1, -1)
        corruption_mask = torch.tensor(corruption_mask, device=device, dtype=torch.float32).repeat([batch_size, num_channels, height, 1])

    return corruption_mask


def ambient_sampler(
    net, latents, randn_like=torch.randn_like,
    num_steps=20, sigma_min=0.01, sigma_max=80, rho=7,
    S_churn=0.0, S_min=0.0, S_max=float('inf'), S_noise=10,
    cond_loc = "",
    image_dir = "",
    cond=None,
    gt_norm=1
    ):
   
    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0    

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]
    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

            # Euler step.
            net_input = torch.cat([x_hat, cond], dim=1)
            denoised = net(net_input, t_hat).to(torch.float64)[:, :1]
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:

                net_input = torch.cat([x_next, cond], dim=1)
                denoised = net(net_input, t_next).to(torch.float64)[:, :1]
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return gt_norm*x_next

def ambient_masked_sampler(
    net, latents, randn_like=torch.randn_like,
    num_steps=20, sigma_min=0.75, sigma_max=80, rho=7,
    S_churn=0.0, S_min=0.0, S_max=float('inf'), S_noise=100,
    num_masks=1, clipping=True, static=True, 
    cond_loc = "",
    image_dir = "",
    cond=None,
    gt_norm=1):

    print("net.sigma_min")
    print(sigma_min)
    print("net.sigma_max")
    print(sigma_max)

    clean_image = None

    def sample_masks():
        masks = []

        for i in range(num_masks):
            corruption_mask = get_well_mask(latents.shape, 3, same_for_all_batch=False, device=latents.device, seed=None)
            masks.append(corruption_mask)

        masks = torch.stack(masks)
        return masks

    masks = sample_masks()

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0  
    
    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1

        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        
        # x_hat = x_cur
        x_hat = x_hat.detach()
        x_hat.requires_grad = True

        denoised = []
        for mask_index in range(num_masks):
            masks = sample_masks()
            corruption_mask = masks[mask_index]
            masked_image = corruption_mask * x_hat

            cond[0,-1,:,:] = corruption_mask

            net_input = torch.cat([masked_image, cond], dim=1)
            net_output = net(net_input, t_hat).to(torch.float64)[:, :1]

            if clipping:
                net_output = tensor_clipping(net_output, static=static)

            if clean_image is not None:
                net_output = corruption_mask * net_output + (1 - corruption_mask) * clean_image

            # Euler step.
            denoised.append(net_output)


        stack_denoised = torch.stack(denoised)
        flattened = stack_denoised.view(stack_denoised.shape[0], -1)
        l2_norm = cdist_masked(flattened, flattened, None, None)
        l2_norm = l2_norm.mean()
        rec_grad = torch.autograd.grad(l2_norm, inputs=x_hat)[0]

        clean_pred = torch.mean(stack_denoised, dim=0, keepdim=True).squeeze(0)
        single_mask_grad = (t_next - t_hat) * (x_hat - clean_pred) / t_hat
        grad_1 = single_mask_grad 
        x_next += grad_1

        if i < num_steps - 1:
            masks = sample_masks()
            x_next = x_next.detach()
            x_next.requires_grad = True

            denoised = []
            for mask_index in range(num_masks):

                corruption_mask = masks[mask_index]
                masked_image = corruption_mask * x_next

                cond[0,-1,:,:] = corruption_mask               

                net_input = torch.cat([masked_image, cond], dim=1)
                net_output = net(net_input, t_hat).to(torch.float64)[:, :1]

                if clipping:
                    net_output = tensor_clipping(net_output, static=static)
                
                if clean_image is not None:
                    net_output = corruption_mask * net_output + (1 - corruption_mask) * clean_image
                denoised.append(net_output)
            
            stack_denoised = torch.stack(denoised)
            flattened = stack_denoised.view(stack_denoised.shape[0], -1)
            l2_norm = cdist_masked(flattened, flattened, None, None)
            rec_grad = torch.autograd.grad(l2_norm, inputs=x_next)[0]

            clean_pred = stack_denoised[0]
            single_mask_grad = (t_next - t_hat) * (x_next - clean_pred) / t_next
            grad_2 = single_mask_grad
            x_next = x_hat + 0.5 * (grad_1 + grad_2)

        else:
            if clean_image is not None:
                x_next = masks[0] * x_next + (1 - masks[0]) * clean_image
            else:
                clean_image = x_next
                x_next = x_hat + grad_1

    return gt_norm * x_next

def main(network_loc, training_options_loc, outdir, seeds, num_steps, max_batch_size, 
         num_generate,  cond_base, back_base, gt_base, gt_norm, cond_norm,use_offsets,out_chan,num_skip, trained_res,c_chan,device=torch.device('cuda'),  **sampler_kwargs):

    seeds = seeds[:num_generate]
    num_batches = ((len(seeds) - 1) // (max_batch_size * 1) + 1) *1
    rank_batches = torch.as_tensor(seeds).tensor_split(num_batches)

    # load training options
    with dnnlib.util.open_url(training_options_loc, verbose=(True)) as f:
        training_options = json.load(f)

    #load in condition
    files_cond = dnnlib.util.list_dir(cond_base)
    cond_loc = cond_base+files_cond[0]

    cond = np.load(cond_loc) / cond_norm
    cond = torch.from_numpy(cond) 
    cond = cond[25, :, :]
    cond = cond.repeat(1,1,1,1).to((device))

    # Add background
    if not (back_base == None):
        back_loc = back_base+"backs_"+files_cond[0][-8:]
        #back_loc = back_base+files_cond[0]
        background = np.load(back_loc)
        background = torch.from_numpy(background)  / gt_norm
        background = background.repeat(1,1,1,1).to((device))
        cond = torch.cat([cond, background], axis=1)

    # Add mask
    if not (back_base == None):
        mask = np.ones((1, 256, 512))
        mask = torch.from_numpy(mask)
        mask = mask.repeat(1,1,1,1).to((device))
        cond = torch.cat([cond, mask], axis=1)

    interface_kwargs = dict(img_resolution=trained_res, label_dim=0, img_channels=cond.shape[1]+1)
    network_kwargs = training_options['network_kwargs']
    model_to_be_initialized = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module

    # find all *.pkl files in the folder network_loc and sort them
    files = dnnlib.util.list_dir(network_loc)
    pkl_files = [f for f in files if f.endswith('.pkl')]

    # Sort the list of "*.pkl" files
    sorted_pkl_files = sorted(pkl_files)
    sorted_pkl_files = [sorted_pkl_files[-1]] # use only the most recent network

    checkpoint_numbers = []
    for curr_file in sorted_pkl_files:
        checkpoint_numbers.append(int(curr_file.split('-')[-1].split('.')[0]))
    checkpoint_numbers = np.array(checkpoint_numbers)

    for checkpoint_number, checkpoint in zip(checkpoint_numbers, sorted_pkl_files):

        network_pkl = os.path.join(network_loc, f'network-snapshot-{checkpoint_number:06d}.pkl')
        # Load network.
        #dist.print0(f'Loading network from "{network_pkl}"...')
        with dnnlib.util.open_url(network_pkl, verbose=True) as f:
            loaded_obj = pickle.load(f)['ema']
        
        if type(loaded_obj) == OrderedDict:
            COMPILE = False
            if COMPILE:
                net = torch.compile(model_to_be_initialized)
                net.load_state_dict(loaded_obj)
            else:
                modified_dict = OrderedDict({key.replace('_orig_mod.', ''):val for key, val in loaded_obj.items()})
                net = model_to_be_initialized
                net.load_state_dict(modified_dict)
        else:
            # ensures backward compatibility for times where net is a model pkl file
            net = loaded_obj
        net = net.to(device)
        #dist.print0(f'Network loaded!')

        ###loop here 
        files_cond = dnnlib.util.list_dir(cond_base)

        ssims = []
        rmses = []
        print(files_cond[0::num_skip])
        for i_str in files_cond[0::num_skip]:

            cond_loc = cond_base+i_str
            gt_loc = gt_base+"gt_"+i_str[-8:]

            gt = np.load(gt_loc) 
            vmin_gt = 1.5
            vmax_gt = np.max(gt)
            cmap_gt = cc.cm['rainbow4']

            cond = np.load(cond_loc) / cond_norm
            cond = torch.from_numpy(cond) 
            cond = cond[25, :, :] # zero-offset
            cond = cond.repeat(1,1,1,1).to((device))

            image_dir = os.path.join(outdir, str(checkpoint_number) + "/" + cond_loc[-13:-4])
            
            os.makedirs(image_dir, exist_ok=True)
            
            if not (back_base == None):
                back_loc = back_base+"backs_"+i_str[-8:]
                background = np.load(back_loc)
                background = torch.from_numpy(background)  / gt_norm
                background = background.repeat(1,1,1,1).to((device))

                cond = torch.cat([cond, background], axis=1)

                mask = np.ones((1, 256, 512))
                mask = torch.from_numpy(mask)
                mask = mask.repeat(1,1,1,1).to((device))
                cond = torch.cat([cond, mask], axis=1)

                plt.figure(); plt.title("Condition back")
                plt.imshow(gt_norm*cond[0,-2,:,:].cpu(), vmin=vmin_gt,vmax=vmax_gt, cmap = cmap_gt)
                plt.axis("off")
                cb = plt.colorbar(fraction=0.0235, pad=0.04); 
                plt.savefig(os.path.join(image_dir, "back_condition.png"),bbox_inches = "tight",dpi=300)

            a = np.quantile(np.absolute(cond[0,0,:,:].cpu()),0.95)
            plt.figure(); plt.title("Condition rtm")
            plt.imshow(cond[0,0,:,:].cpu(), vmin=-a,vmax=a, cmap = "gray")
            plt.axis("off")
            cb = plt.colorbar(fraction=0.0235, pad=0.04); 
            plt.savefig(os.path.join(image_dir, "rtm_condition.png"),bbox_inches = "tight",dpi=300)

            plt.figure();  plt.title("Ground truth")
            plt.imshow(gt, cmap = cmap_gt)
            plt.axis("off")
            cb = plt.colorbar(fraction=0.0235, pad=0.04); cb.set_label('[Km/s]')
            plt.savefig(os.path.join(image_dir, "original_velocity.png"),bbox_inches = "tight",dpi=300)


            # Loop over batches.
            #dist.print0(f'Generating {len(seeds)} images to "{outdir}"...')
            batch_count = 1
            images_np_stack = np.zeros((len(seeds),1,*gt.shape))
            for batch_seeds in tqdm.tqdm(rank_batches):
                batch_size = len(batch_seeds)
                if batch_size == 0:
                    continue

                # Pick latents and labels.
                rnd = StackedRandomGenerator(device, batch_seeds)
                latents = rnd.randn([batch_size, 1, gt.shape[0], gt.shape[1]], device=device)
               
                # Generate images.
                sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
                images = ambient_masked_sampler(net, latents,num_steps=num_steps, randn_like=rnd.randn_like,
                    cond=cond, image_dir=image_dir,gt_norm=gt_norm, **sampler_kwargs)
                
                # Save Images
                images_np = images.cpu().detach().numpy()
                for seed, one_image in zip(batch_seeds, images_np):
                    #dist.print0(f"Saving loc: {image_dir}")
                    os.makedirs(image_dir, exist_ok=True)
                    image_path = os.path.join(image_dir, "steps_"+str(num_steps)+"_"+f'{seed:04d}.png')

                    plt.figure(); plt.title("Posterior Sample")
                    plt.imshow(one_image[0, :, :],   vmin=vmin_gt,vmax=vmax_gt,cmap = cmap_gt)
                    plt.axis("off")
                    cb = plt.colorbar(fraction=0.0235, pad=0.04); cb.set_label('[Km/s]')
                    plt.savefig(image_path, bbox_inches = "tight",dpi=300)
                    plt.close()
                    os.makedirs(os.path.join(image_dir, f'saved/'), exist_ok=True)
                    np.save(os.path.join(image_dir, f'saved/{seed:06d}')+ ".npy", one_image[0, :, :])
                images_np_stack[batch_count-1,0,:,:] = one_image
                batch_count += 1

            # plot posterior statistics
            post_mean = np.mean(images_np_stack,axis=0)[0,:,:]
            ssim_t = ssim(gt,post_mean, data_range=np.max(gt) - np.min(gt))

            plt.figure(); plt.title("Posterior mean SSIM:"+str(round(ssim_t,4)))
            plt.imshow(post_mean,  vmin=vmin_gt,vmax=vmax_gt,   cmap = cmap_gt)
            np.save(os.path.join(image_dir, f'saved/posterior_mean')+ ".npy", post_mean)
            plt.axis("off"); 
            cb = plt.colorbar(fraction=0.0235, pad=0.04); cb.set_label('[Km/s]')
            plt.savefig(os.path.join(image_dir, "steps_"+str(num_steps)+"_num_"+str(num_generate)+"_mean.png"),bbox_inches = "tight",dpi=300); plt.close()

            plt.figure(); plt.title("Stdev")
            plt.imshow(np.std(images_np_stack,axis=0)[0,:,:],   cmap = "magma")
            np.save(os.path.join(image_dir, f'saved/posterior_std')+ ".npy", np.std(images_np_stack,axis=0)[0,:,:])
            plt.axis("off"); plt.colorbar(fraction=0.0235, pad=0.04)
            plt.savefig(os.path.join(image_dir, "steps_"+str(num_steps)+"_num_"+str(num_generate)+"std.png"),bbox_inches = "tight",dpi=300); plt.close()
                
            rmse_t = np.sqrt(mean_squared_error(gt, post_mean))
            plt.figure(); plt.title("Error RMSE:"+str(round(rmse_t,4)))
            plt.imshow(np.abs(post_mean-gt), vmin=0, vmax=0.5, cmap = "magma")
            plt.axis("off"); plt.colorbar(fraction=0.0235, pad=0.04)
            plt.savefig(os.path.join(image_dir, "steps_"+str(num_steps)+"_num_"+str(num_generate)+"_error.png"),bbox_inches = "tight",dpi=300); plt.close()

            ssims.append(ssim_t)
            rmses.append(rmse_t)

            print("SSIM:"+str(ssim_t))
            print("rmses:"+str(rmse_t))

            # Ensure the metrics directory exists
            metrics_dir = os.path.join(outdir, "metrics")
            os.makedirs(metrics_dir, exist_ok=True)
            
            np.save(os.path.join(outdir,"metrics/",f'{seed:06d}')+ "_ssims.npy", ssims)
            np.save(os.path.join(outdir,"metrics/",f'{seed:06d}')+ "_rmses.npy", rmses)

        np.save(os.path.join(outdir,"metrics/", f'{seed:06d}')+ "_ssims.npy", ssims)
        np.save(os.path.join(outdir,"metrics/", f'{seed:06d}')+ "_rmses.npy", rmses)
    
if __name__ == "__main__":
   
    seeds = [i for i in range(0, 100)]
    use_offsets = False
    out_chan = 1
    num_skip = 1

    max_batch_size = 1
    num_steps = 20

    device = torch.device('cuda')

    parser = argparse.ArgumentParser()
    parser.add_argument('--cond_loc', type=str, default="")
    parser.add_argument('--back_loc', type=str, default=None)
    parser.add_argument('--network_loc', type=str, default="")
    parser.add_argument('--gt_loc', type=str, default="")
    parser.add_argument('--gt_norm', type=float, default=1.0)
    parser.add_argument('--cond_norm', type=float, default=1.0)
    parser.add_argument('--out_chan', type=int, default=1)
    parser.add_argument('--c_chan', type=int, default=1)
    parser.add_argument('--num_skip', type=int, default=1)
    parser.add_argument('--trained_res', type=int, default=256)
    parser.add_argument('--num_generate', type=int, default=16)

    args = parser.parse_args()
    cond_loc = args.cond_loc
    back_loc = args.back_loc
    vel_loc = args.gt_loc
    network_loc = args.network_loc
    gt_norm = args.gt_norm
    cond_norm = args.cond_norm
    out_chan = args.out_chan
    c_chan = args.c_chan
    num_skip = args.num_skip
    trained_res = args.trained_res
    num_generate = args.num_generate

    training_options_loc = network_loc+"/training_options.json"
    outdir = f"samples/"

    main(network_loc, training_options_loc, outdir, seeds, num_steps, max_batch_size, 
         num_generate,  cond_loc,back_loc, vel_loc, gt_norm, cond_norm, use_offsets,out_chan,num_skip,trained_res,c_chan,device)
    