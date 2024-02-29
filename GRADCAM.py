# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 13:20:28 2023

@author: uvl
"""
import numpy as np
import os
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.data.sampler import RandomSampler
from torchvision import datasets, models, transforms
import torchvision
from multiprocessing import cpu_count
from sklearn.metrics import classification_report, cohen_kappa_score
from collections import Counter, OrderedDict, defaultdict
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pytorch_grad_cam import GradCAM
from models import Effnet_Melanoma
from SA_models import AttnEfficientNetB3
from GradCamdataset import get_df, get_transforms, MelanomaDataset
from functools import reduce

os.environ['CUDA_VISIBLE_DEVICES'] = 'MIG-be382572-3ee9-5cd0-8100-e4e9ff297d7a'

device = torch.device('cuda')
# =============================================================================
# load your model
# =============================================================================
model_dir = '/home/falcon/sana/SIIM-ISIC/test_weights/train3'
ModelClass = Effnet_Melanoma
kernel_type = '8c_b3_224_224_18ep'
enet_type = 'tf_efficientnet_b3_ns'
out_dim = 8
fold = 0 
    # load model
models = []
follds= [0]
for fold in follds:
    model_file = os.path.join(model_dir, f'{kernel_type}_best_fold{fold}.pth')
    model = ModelClass(
        enet_type=enet_type,
        out_dim=out_dim
    )
    model = model.to(device)

    try:  # single GPU model_file
        model.load_state_dict(torch.load(model_file), strict=True)
    except:  # multi GPU model_file
        state_dict = torch.load(model_file)
        state_dict = {k[7:] if k.startswith('module.') else k: state_dict[k] for k in state_dict.keys()}
        model.load_state_dict(state_dict, strict=True)
    
    model.eval()
    models.append(model)
# =============================================================================
# load your data
# =============================================================================
image_size = 512
batch_size= 9

df, mel_idx = get_df(
        '8c_b3_224_224_18ep', 8, './data/',512, use_meta= False)
transforms_train, transforms_val = get_transforms(image_size)   
dataset_train = MelanomaDataset(df, 'train', transform=transforms_val) 
train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, num_workers=4)

with torch.no_grad():
    model_layers = nn.Sequential(model.enet.conv_stem, model.enet.bn1, model.enet.act1,model.enet.blocks,model.enet.conv_head, model.enet.bn2, model.enet.act2)
    #model_layers = nn.Sequential(model.conv_stem, model.bn1, model.act1,model.blocks,model.conv_head, model.bn2, model.act2)

class_names = ['AK', 'BCC','BKL','DF','SCC','VASC','MEL','NV']
###############################################################################
def visualize(model, dataloader, num_images=2):
    was_training = model.training
    model.eval()
    model.to(device)
    images_so_far = 0

    cam = GradCAM(model=model, target_layers=model_layers, use_cuda=False)
    
    fig = plt.figure(constrained_layout=True, figsize=(48,24))
    fig.tight_layout()
    subfigs = fig.subfigures(nrows=num_images, ncols=1)

    for inputs, labels in dataloader:      
        inputs = inputs.to(device)                      # B x C x W x H
        labels = labels.to(device)
            
        outputs = model(inputs)                         # B x classes
        prob = nn.functional.softmax(outputs, dim=-1)
        prob, idx = prob.sort(1,True) 
        # B x classes
        print("y_true:",labels) 
        #print("y_pred:", prob)
        print("isx", idx)
        masks = cam(input_tensor=inputs[:num_images])
        print("input size",inputs.size()[0])
        for j in range(inputs.size()[0]):
            axs = subfigs[j].subplots(1, 2)
            subfigs[j].suptitle('Predicted {1}/{2} \nwith confidence {0:.4f}'.format(prob[j][0],
                class_names[idx[j][0]].capitalize(), class_names[labels[j]].capitalize()))
            axs[0].axis('off')
            axs[0].set_title('Input')  
            
            #torchvision.utils.save_image(inputs[j], './GradCAM_data/figs/real_image.png')   
            img = inputs[j].cpu().squeeze().permute(2, 1, 0).detach().numpy()
            axs[0].imshow(img)
            axs[1].axis('off')

            gradcam_output_normalized = (masks[j] * 255).astype(np.uint8)

# Transpose the GradCAM output array to align it correctly
            masks[j] = np.transpose(gradcam_output_normalized)
            axs[1].set_title('Grad-CAM')
            axs[1].imshow(masks[j])
            """
            imask = Image.fromarray((masks[j] * 255).astype(np.uint8))
            mask_rgb = cv2.cvtColor(imask, cv2.COLOR_BGR2RGB)
            mask_rgb.save('./GradCAM_data/figs/mask_image.png')
            """
            images_so_far += 1
            if images_so_far == num_images:            
                model.train(mode=was_training)
                fig.savefig(os.path.join('./GradCAM_data/figs/','gradcam_figure.png'))
                print("gradcam figure saved")
                return
                

    model.train(mode=was_training)

############################visualize activation###################################################################
def visualize_activations(convs, dataloader):
    for inputs, labels in dataloader:
        inputs = inputs.to(device)  # B x C x H x W
        img = inputs[:1]

        activations = [convs[0](img)]
        for i in range(1, len(convs)):
            activations.append(convs[i](activations[-1]))
        counter = 0
        for conv, act in zip(convs, activations):
            out_ch = conv.out_channels
            fig = plt.figure(figsize=(out_ch*2, 2), constrained_layout=True)
            fig.suptitle(f'CONV: {conv}', size=16)
            for j in range(out_ch):
                ax = plt.subplot(1, out_ch, j+1)
                ax.axis('off')
                ax.imshow(act.detach().squeeze()[j].cpu(), cmap='gray')
                # Dim --> B x Out Ch x H' x W'
            fig.savefig(f'./GradCAM_data/figs/activation_figure{counter}.png')          
            counter+= 1
            print("activation figures saved!")
            plt.close(fig)
        return

 
def get_convs(layers):
    conv_layers = []
    for module in layers:
        if 'Conv2d' in str(type(module)):
            conv_layers.append(module)
        elif len(list(module.children()))>1:
            conv_layers.extend(get_convs(module.children()))
    return conv_layers


with torch.no_grad():
    conv_layers = nn.Sequential(model.enet.conv_stem, model.enet.blocks,model.enet.conv_head) 
    #conv_layers = nn.Sequential(model.conv_stem, model.blocks,model.conv_head)

    
visualize(model, train_loader)
convs = get_convs(conv_layers)
#visualize_activations(convs, train_loader)
