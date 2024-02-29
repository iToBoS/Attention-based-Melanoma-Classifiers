# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 13:20:28 2023

@author: uvl
"""
import numpy as np
import os
import torch
import torch.nn as nn
from torchvision import  models, transforms
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from pytorch_grad_cam import GradCAM
import zennit
from zennit.composites import LayerMapComposite, EpsilonPlusFlat
from zennit.rules import Epsilon, ZPlus, Pass, Flat
from zennit.composites import DeconvNet, GuidedBackprop, ExcitationBackprop
from ensemble_attn import load_models

data_dir = './GradCAM_data/files/'

##############################################################################
def predict_single(model, x, detach=False):
      '''
            receive a torch tensor x, a model and some optional stuff to make single decision.
      '''
      prediction = model.forward(x.unsqueeze(0))  # package singular tensor x into 1-sized batch.
      prediction = prediction[0]                  # unpack again

      if detach: prediction = prediction.detach()

      ypred = int(torch.argmax(prediction).detach().cpu().numpy())
      return prediction, ypred
############################visualize ###################################################################
def visualize_attr(attr,path,  norm='abs', clamp_neg=False, aggregate_channels='sum', cmap='seismic', text=None, color_level=0.9, show=False):
      # expects a tensor attr of shape (C x H x W), ie one value per voxel.
      attr = attr.detach().cpu().numpy()
      if clamp_neg:
            attr = np.maximum(attr, 0)
      if aggregate_channels: # sum, abs_sum, l2
            # assumes the existance of channel axis at tensor axis index 0
            if aggregate_channels == 'sum':
                  attr = attr.sum(axis=0)
            elif aggregate_channels == 'abs_sum':
                  attr = np.abs(attr).sum(axis=0)
            elif aggregate_channels == 'l2':
                  attr = np.sqrt((attr*attr).sum(0))
      if norm == 'abs':
            attr /= np.abs(attr).max() # attr to range [-1,1]
            attr = (attr + 1)/2        # attr to range [0,1]

      # colorize attributions. assumes input values within [0,1].
      cmap = matplotlib.colormaps[cmap]
      attr_shape = attr.shape
      attr_img = cmap(np.reshape(attr, [np.prod(attr.shape)])) # returns RGBA
      attr_img = np.reshape(attr_img[...,0:3], attr_shape + (3,))# reshape RGB information (only) to original attribution shape
      attr_img = (attr_img * color_level * 255).astype(np.uint8)

      if show:
          #plt.imshow(attr_img, cmap=cmap, vmin=0, vmax=1)
          if text: plt.title(text)
          #plt.show()
          plt.savefig(os.path.join(path, f'{text}_plot.png'))
      
      return attr_img
############################visualize ###################################################################
def show_images(imgs,path,  title='Inputs'):
    fig, axes = plt.subplots(1, len(imgs), figsize=(15, 3), squeeze=False)
    for i in range(len(imgs)):
      ax = axes[0,i]
      ax.imshow(imgs[i])
      ax.axis('off')
      if i == 0 and title:
        ax.set_title(title)
    #plt.show()
    plt.savefig(os.path.join(path, f'{title}_plot.png'))
############################visualize ###################################################################
# display the attributions. the **kwargs are passed to the visualization helper viszalize_attr
def show_attributions(attrs, path, title=None, **kwargs):
    fig, axes = plt.subplots(1, len(attrs), figsize=(15, 3), squeeze=False)
    for i in range(len(attrs)):
      ax = axes[0,i]
      ax.imshow(visualize_attr(attrs[i],path=plots_dir, **kwargs))
      ax.axis('off')
      if i == 0 and title:
        ax.set_title(title)
    #plt.show()
    plt.savefig(os.path.join(path, f'{title}_plot.png'))
############################gradients ###################################################################
def xai_gradient(model, x, target=None):
  x.requires_grad=True

  y_pred, y = predict_single(model, x)
  if not target: target = y

  grad, = torch.autograd.grad(y_pred[target], x, y_pred[target]) 
  return grad, target

def xai_grad_times_input(model, x, target=None):
  grad, target = xai_gradient(model, x, target) 
  return x*grad, target 
def xai_sensitivity(model, x, target=None):
  grad, target = xai_gradient(model, x, target) 
  return torch.sqrt(grad**2), target 

def xai_smoothgrad(model, x, n=20, std=1, target=None):
  sgrad, target = xai_gradient(model, x, target) 
  if n > 1: 
    for i in tqdm.tqdm(range(1,n), desc='Smoothgrad iterations'): 
      sgrad += xai_gradient(model, torch.clone(x.detach()) + torch.randn_like(x) * std, target)[0] 
    sgrad /= n

  return sgrad, target
############################GRADCAM ###################################################################
def gc_single_image(model,image,label,title,path):
    was_training = model.training
    model.eval()
    model.to(device)
    cam = GradCAM(model=model, target_layers=model_layers, use_cuda=False)

    # Calculate probabilities and indices
    prob = 0
    for i in range(8):
      output = model(image.unsqueeze(0).to(device))
      prob += nn.functional.softmax(output, dim=-1)
    prob /= 8
    prob, idx = prob.sort(1, True)

    fig, axs = plt.subplots(1, 2, figsize=(24, 12))
    fig.suptitle('Predicted {1} \nwith confidence {0:.4f}'.format(prob[0][0],
                class_names[idx[0][0]].capitalize(), class_names[label].capitalize()))

    axs[0].axis('off')
    axs[0].set_title('Input')

    img = image.cpu().squeeze().permute(2, 1, 0).detach().numpy()
    axs[0].imshow(img)
    axs[1].axis('off')

    mask = cam(input_tensor=image.unsqueeze(0))
    gradcam_output_normalized = (mask[0] * 255).astype(np.uint8)

    # Transpose the GradCAM output array to align it correctly
    mask = np.transpose(gradcam_output_normalized)
    axs[1].set_title('Grad-CAM')
    axs[1].imshow(mask)

    model.train(mode=was_training)
    #plt.show()
    plt.savefig(os.path.join(path, f'{title}_plot.png'))

####################################################################################################################
def xai_zennit(model, x, RuleComposite, target=None):
  # this "with"-context places backward hooks (temporarily) replacing / modifying the vanilla gradient backward pass using the passed RuleComposite.
  with RuleComposite.context(model):
    x.requires_grad = True
    y_pred, y = predict_single(model, x)
    if not target: target = y
    # initiate relevance
    relevance_init = y_pred *  torch.eye(np.prod(y_pred.shape),device=device)[target]
    relevance, = torch.autograd.grad(y_pred, x, relevance_init)
  return relevance, target

#################################################################################################################### 
def get_convs(layers):
    conv_layers = []
    for module in layers:
        if 'Conv2d' in str(type(module)):
            conv_layers.append(module)
        elif len(list(module.children()))>1:
            conv_layers.extend(get_convs(module.children()))
    return conv_layers

os.environ['CUDA_VISIBLE_DEVICES'] = 'MIG-be382572-3ee9-5cd0-8100-e4e9ff297d7a'
device = torch.device('cuda')
# =============================================================================
# load your model
# =============================================================================
print("Loading models...")
models = load_models(folds=[1], add_base=True)
out_dim = 8
# =============================================================================
# load your data
# =============================================================================
image_size = 384
batch_size= 9
class_names = ['AK', 'BCC','BKL','DF','SCC','VASC','MEL','NV']
images = []
image_names = []
print("==================================================== \n \n \n \n YOU MUST ADD LABELS OF YOUR IMAGES HERE MANUALLY \n \n \n \n ====================================================== ")

#labels = [0,2,4,6,7,5,6,3,1,1]
labels = [6,6,6,6]
#['ISIC_2829582', 'ISIC_3491559', 'ISIC_4799999', 'ISIC_4223497', 'ISIC_9262713', 'ISIC_1935453', 'ISIC_4615762', 'ISIC_5377242', 'ISIC_6377613', 'ISIC_5206639']
data_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),  
    transforms.ToTensor()  ])

# iterate over all files in data dir
for file in os.listdir(data_dir):
    if file.endswith((".jpg", ".JPG", ".png")):
        file_path = os.path.join(data_dir, file)
        file_name, file_extension = os.path.splitext(file)
        image_names.append(file_name)
        image = Image.open(file_path).convert('RGB')
        images.append(image)
        #input_tensor = data_transform(image).unsqueeze(0)
print(f"Found {len(images)} images in the data folder.") 
print(image_names)
model_names = ["CBAM_S2_B3", "SGE_B3", "S2_B3_2", "Spatial_B3","baseline"]

for ii in range(len(model_names)):
    print(f"Generating attributions for {model_names[ii]}...")
    model = models[ii]
    plots_dir = f'./GradCAM_data/plots/{model_names[ii]}/'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    if model_names[ii] == "baseline":
        with torch.no_grad():
            model_layers = nn.Sequential(model.enet.conv_stem, model.enet.bn1, model.enet.act1,*model.enet.blocks,model.enet.conv_head, 
                                        model.enet.bn2, model.enet.act2) 
            conv_layers = nn.Sequential(model.enet.conv_stem, model.enet.blocks,model.enet.conv_head) 
    else:
      with torch.no_grad():
          model_layers = nn.Sequential(model.conv_stem, model.bn1, model.act1,*model.blocks,model.conv_head, 
                                      model.bn2, model.act2) 
          conv_layers = nn.Sequential(model.conv_stem, model.blocks,model.conv_head)

    for i in tqdm.tqdm(range(len(images))):
    # =============================================================================
        plots_dir = f'./GradCAM_data/plots/{model_names[ii]}/{image_names[i]}/'
        if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)
        image_transformed = data_transform(images[i]).to(device)
                     
        """        
        grad_attrs = [xai_gradient(model, image_transformed, target=target)[0] for target in [None, labels[i]]]
        show_attributions(grad_attrs,path=plots_dir, title=f'Gradients_{image_names[i]}')

        sens_attrs = [xai_sensitivity(model, image_transformed, target=target)[0] for target in [None, labels[i]]]
        show_attributions(sens_attrs,path=plots_dir, title=f'Sensetivity_{image_names[i]}', aggregate_channels='l2')

        gxi_attrs = [xai_grad_times_input(model, image_transformed, target=target)[0] for target in [None, labels[i]]]
        show_attributions(gxi_attrs,path=plots_dir, title=f'Gradxinput_{image_names[i]}')
        n = 10
        std = 1
        sg_attrs = [xai_smoothgrad(model, image_transformed, n=n, std=std, target=target)[0] for target in [None, labels[i]]]
        show_attributions(sg_attrs,path=plots_dir, title=f'SG_({n},{std})_{image_names[i]}', aggregate_channels='abs_sum')
        """
            
        gc_single_image(model, image_transformed, labels[i], title=f'GRADCAMS__{image_names[i]}',path=plots_dir)
        # =============================================================================
        layer_map = [
            (nn.SiLU, Pass()),     
            (nn.Conv2d, ZPlus()), 
            (nn.Linear, Epsilon(epsilon=1e-6)) ]
        """
        attr_znt = [xai_zennit(model, image_transformed, RuleComposite=LayerMapComposite(layer_map=layer_map), target=t)[0] for t in [None, labels[i]]] 
        show_attributions(attr_znt,path=plots_dir, title=f'LRP-E+_{image_names[i]}')

        
              
        attr_znt = [xai_zennit(model, image_transformed, RuleComposite=EpsilonPlusFlat(), target=t)[0] for t in [None, labels[i]]] 
        show_attributions(attr_znt,path=plots_dir, title=f'LRP-E+F_{image_names[i]}')
        """

        attr_znt = [xai_zennit(model, image_transformed, RuleComposite=GuidedBackprop(), target=t)[0] for t in [None,  labels[i]]] # 282 is "tiger cat"
        show_attributions(attr_znt,path=plots_dir, title=f'GBP_{image_names[i]}')
        """
        attr_znt = [xai_zennit(model, image_transformed, RuleComposite=ExcitationBackprop(), target=t)[0] for t in [None,  labels[i]]] # 282 is "tiger cat"
        show_attributions(attr_znt,path=plots_dir, title=f'EBP_{image_names[i]}')
        """