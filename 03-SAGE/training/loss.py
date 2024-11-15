# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import torch
from torch_utils import persistence
import numpy as np

#----------------------------------------------------------------------------
# EDMLoss for Ambient Diffusion
@persistence.persistent_class
class ConditionalLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, norm=2):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.norm = norm

    def __call__(self, net, images, cond, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        
        noisy_image = y + n
        cat_input = torch.cat([noisy_image, cond], axis=1)
        D_yn = net(cat_input, sigma,  augment_labels=augment_labels)
        #print(y.shape)
        #print(D_yn.shape)

        #D_yn = D_yn[:, :3]
        #print(D_yn.shape)
        #print((D_yn - y).shape)
        
        train_loss = weight * ((D_yn - y) ** 2)
         
        return train_loss
    
#----------------------------------------------------------------------------
# EDMLoss for Ambient Diffusion
@persistence.persistent_class
class AmbientLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, norm=2):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.norm = norm

    def __call__(self, net, images, cond, corruption_matrix, hat_corruption_matrix, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        
        masked_image = hat_corruption_matrix * (y + n)

        cat_input = torch.cat([masked_image, cond, hat_corruption_matrix], axis=1)
        D_yn = net(cat_input, sigma, augment_labels=augment_labels)
        
        train_loss = weight * ((hat_corruption_matrix * (D_yn - y)) ** 2)
        val_loss = weight * ((corruption_matrix * (D_yn - y)) ** 2)
    
        return train_loss, val_loss
   