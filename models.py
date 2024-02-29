import torch
import torch.nn as nn
import geffnet
from resnest.torch import resnest101
from pretrainedmodels import se_resnext101_32x4d
import einops
import torchvision.models as models
import timm
import torch.nn.functional as F
from SA_models_multi import CBAMBlock , S2Attention,  SpatialGroupEnhance, My_SpatialAttention
sigmoid = nn.Sigmoid()

#activation function
class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))
class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)
class Effnet_Melanoma(nn.Module):
    def __init__(self, enet_type='tf_efficientnet_b3', out_dim=8, n_meta_features=0, n_meta_dim=[512, 128], 
                 pretrained=False,freeze_top=False,Test=False):
        super(Effnet_Melanoma, self).__init__()
        self.n_meta_features = n_meta_features
        if 'efficientnetv2' in enet_type:  #'tf_efficientnetv2_s'
            self.enet =  timm.create_model(enet_type, pretrained=pretrained)
        
        else:
            self.enet = geffnet.create_model(enet_type, pretrained=pretrained)
        if Test:
            dp= 0
        else:
            dp=0
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(dp) #was 5 dropouts
        ])
        in_ch = self.enet.classifier.in_features

        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),               
                Swish_Module(),
                nn.BatchNorm1d(n_meta_dim[0]),
                nn.Dropout(p=0.3), #was 0.3
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                Swish_Module(),
                nn.BatchNorm1d(n_meta_dim[1]),
            )
            in_ch += n_meta_dim[1]
        self.myfc = nn.Linear(in_ch, out_dim)
        self.enet.classifier = nn.Identity() 
        if freeze_top:
            for param in self.enet.parameters():
                param.requires_grad = False
        #The classifier module in EfficientNet is typically used as the final fully-connected layer that produces the output predictions. By setting it to an
        # identity function using nn.Identity(), you are effectively removing this final fully-connected layer from the model.
        #This means that the output of the EfficientNet model will be the features extracted from the last convolutional layer, rather than the final predictions.

    def extract(self, x):
        x = self.enet(x)
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        if len(self.dropouts)!=0:
            out /= len(self.dropouts)
        else:
            out= self.myfc(x)
        return out   
class AttentionLayer_ch(nn.Module):
    def __init__(self, in_ch):
        super(AttentionLayer_ch, self).__init__() 
        #p = ((stride - 1) * size of input image - stride + filter size) / 2
        self.layer1 = nn.Conv2d(in_channels=in_ch, out_channels=64, kernel_size=1, stride=1, padding=0, bias=False)
        #swish
        self.layer2 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=1, stride=1, padding=0, bias=False)
        #swish
        #self.layer_3 = LocallyConnected2d(in_channels=16, out_channels=1, output_size=20, kernel_size=1, stride=1, bias=True)
        self.layer3 = nn.Conv2d(in_channels=16,out_channels =8, kernel_size=1, stride=1, padding=0, bias= False)
        self.layer4 = nn.Conv2d(in_channels=8,out_channels =1, kernel_size=1, stride=1, padding=0, bias= False)
        #sigmoid   
        self.layer5 = nn.Conv2d(in_channels=1, out_channels=in_ch, kernel_size=1, stride=1, padding=0, bias=False)
        up_c2_w = torch.ones((in_ch, 1 ,1, 1), dtype=torch.float32)
        self.layer5.weight= nn.Parameter(up_c2_w,requires_grad=False)    
        
    def forward(self, x):
        weights = F.silu(self.layer1(x)) 
        weights = F.silu(self.layer2(weights))
        weights = F.silu(self.layer3(weights))
        weights= sigmoid(self.layer4(weights))        
        weights = self.layer5(weights) #final weights 

        x = x * weights  ##1   
        gap =nn.AdaptiveAvgPool2d((1, 1)) 
        x  = gap(x)
        weights= gap(weights) 
        x = x / (weights + 1e-8)  # Adding a small epsilon for numerical stability
        return x

class S2MLP_B3(nn.Module):
    def __init__(self,  enet_type='tf_efficientnet_b3', out_dim=8, pretrained=False):
        super(S2MLP_B3, self).__init__()
        self.enet = geffnet.create_model(enet_type, pretrained=pretrained)
        in_ch = self.enet.classifier.in_features  
        self.conv_stem = self.enet.conv_stem 
        self.act1 = self.enet.act1
        self.bn1 = self.enet.bn1
        self.blocks = nn.ModuleList()
        self.blocks.append(self.enet.blocks[0])
        attn_ich = self.enet.blocks[0][0].conv_pw.out_channels
        self.blocks.append(self.enet.blocks[1])
        attn_ich = self.enet.blocks[1][0].conv_pwl.out_channels
        self.blocks.append(self.enet.blocks[2])
        attn_ich = self.enet.blocks[2][0].conv_pwl.out_channels
        self.blocks.append(self.enet.blocks[3])
        attn_ich = self.enet.blocks[3][0].conv_pwl.out_channels
        self.blocks.append(self.enet.blocks[4])
        attn_ich = self.enet.blocks[4][0].conv_pwl.out_channels
        self.blocks.append(self.enet.blocks[5]) 
        attn_ich = self.enet.blocks[5][0].conv_pwl.out_channels
        self.blocks.append(S2Attention(channels=attn_ich))
        self.blocks.append(self.enet.blocks[6]) 
        attn_ich = self.enet.blocks[6][0].conv_pwl.out_channels         
        self.conv_head =self.enet.conv_head 
        self.bn2 = self.enet.bn2
        self.act2 = self.enet.act2
        self.global_pool = self.enet.global_pool 
        attn_ich = self.conv_head.out_channels  
        self.classifier = nn.Linear(in_ch, out_dim)
        #self.classifier2 = nn.Linear(out_dim, 2) 
        self.enet = nn.Identity()
   
    def extract(self, x):    
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        for block in self.blocks:
            x = block(x) 
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.global_pool(x)             
        return x
    def forward(self, x, x_meta=None):
        x = self.extract(x)
        x= x.squeeze(-1).squeeze(-1)  
        out1 = self.classifier(x)         
        return out1
class S2MLP_CBAM(nn.Module):
    def __init__(self,  enet_type='tf_efficientnet_b3', out_dim=8, pretrained=False):
        super(S2MLP_CBAM, self).__init__()
        self.enet = geffnet.create_model(enet_type, pretrained=pretrained)
        in_ch = self.enet.classifier.in_features  
        self.conv_stem = self.enet.conv_stem 
        self.act1 = self.enet.act1
        self.bn1 = self.enet.bn1
        self.blocks = nn.ModuleList()
        self.blocks.append(self.enet.blocks[0])

        attn_ich = self.enet.blocks[0][0].conv_pw.out_channels
        self.blocks.append(CBAMBlock(attn_ich))
        self.blocks.append(self.enet.blocks[1])
        attn_ich = self.enet.blocks[1][0].conv_pwl.out_channels
        self.blocks.append(CBAMBlock(attn_ich))
        self.blocks.append(self.enet.blocks[2])
        attn_ich = self.enet.blocks[2][0].conv_pwl.out_channels
        self.blocks.append(CBAMBlock(attn_ich))
 
        self.blocks.append(self.enet.blocks[3])
        attn_ich = self.enet.blocks[3][0].conv_pwl.out_channels
        self.blocks.append(CBAMBlock(attn_ich))
        self.blocks.append(self.enet.blocks[4])
        attn_ich = self.enet.blocks[4][0].conv_pwl.out_channels
        self.blocks.append(CBAMBlock(attn_ich))
        self.blocks.append(self.enet.blocks[5])

        attn_ich = self.enet.blocks[5][0].conv_pwl.out_channels
        self.blocks.append(S2Attention(channels=attn_ich))
        self.blocks.append(CBAMBlock(attn_ich))

        self.blocks.append(self.enet.blocks[6]) 
        attn_ich = self.enet.blocks[6][0].conv_pwl.out_channels   
        self.blocks.append(CBAMBlock(attn_ich))        
        self.conv_head =self.enet.conv_head 
        self.bn2 = self.enet.bn2
        self.act2 = self.enet.act2
        self.global_pool = self.enet.global_pool 
        attn_ich = self.conv_head.out_channels  
        self.classifier = nn.Linear(in_ch, out_dim)
        self.enet = nn.Identity()
   
    def extract(self, x):    
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        for block in self.blocks:
            x = block(x) 
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.global_pool(x)            
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x)
        x= x.squeeze(-1).squeeze(-1)  
        out1 = self.classifier(x)     
        return out1
class SGE_B3(nn.Module):
    def __init__(self,  enet_type='tf_efficientnet_b3', out_dim=8, pretrained=False):
        super(SGE_B3, self).__init__()
        self.enet = geffnet.create_model(enet_type, pretrained=pretrained)
        in_ch = self.enet.classifier.in_features  
        self.conv_stem = self.enet.conv_stem 
        self.act1 = self.enet.act1
        self.bn1 = self.enet.bn1
        self.blocks = nn.ModuleList()
        self.blocks.append(self.enet.blocks[0])
        self.blocks.append(self.enet.blocks[1])
        attn_ich = self.enet.blocks[1][0].conv_pwl.out_channels
        self.blocks.append(SpatialGroupEnhance(groups=8))
        self.blocks.append(self.enet.blocks[2])
        attn_ich = self.enet.blocks[2][0].conv_pwl.out_channels
        self.blocks.append(SpatialGroupEnhance(groups=8))
        self.blocks.append(self.enet.blocks[3])
        attn_ich = self.enet.blocks[3][0].conv_pwl.out_channels
        self.blocks.append(SpatialGroupEnhance(groups=8))
        self.blocks.append(self.enet.blocks[4])
        attn_ich = self.enet.blocks[4][0].conv_pwl.out_channels
        self.blocks.append(SpatialGroupEnhance(groups=8))
        self.blocks.append(self.enet.blocks[5]) 
        attn_ich = self.enet.blocks[5][0].conv_pwl.out_channels
        self.blocks.append(SpatialGroupEnhance(groups=8))
        self.blocks.append(self.enet.blocks[6]) 
        attn_ich = self.enet.blocks[6][0].conv_pwl.out_channels   
        self.blocks.append(SpatialGroupEnhance(groups=8))        
        self.conv_head =self.enet.conv_head 
        self.bn2 = self.enet.bn2
        self.act2 = self.enet.act2
        self.global_pool = self.enet.global_pool 
        attn_ich = self.conv_head.out_channels  
        self.classifier = nn.Linear(in_ch, out_dim)
        #self.classifier2 = nn.Linear(out_dim, 2) #for binary classification
        self.enet = nn.Identity()
   
    def extract(self, x):    
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        for block in self.blocks:
            x = block(x) 
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.global_pool(x)             
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x)
        x= x.squeeze(-1).squeeze(-1)  
        out1 = self.classifier(x)      
        return out1
class Spatial_B3(nn.Module):
    def __init__(self,  enet_type='tf_efficientnet_b3', out_dim=8, pretrained=False):
        super(Spatial_B3, self).__init__()
        self.enet = geffnet.create_model(enet_type, pretrained=pretrained)
        in_ch = self.enet.classifier.in_features  
        self.conv_stem = self.enet.conv_stem 
        self.act1 = self.enet.act1
        self.bn1 = self.enet.bn1
        self.blocks = nn.ModuleList()
        self.blocks.append(self.enet.blocks[0])
        self.blocks.append(self.enet.blocks[1])
        attn_ich = self.enet.blocks[1][0].conv_pwl.out_channels
        self.blocks.append(My_SpatialAttention())        
        self.blocks.append(self.enet.blocks[2])
        attn_ich = self.enet.blocks[2][0].conv_pwl.out_channels
        self.blocks.append(My_SpatialAttention())
        self.blocks.append(self.enet.blocks[3])
        attn_ich = self.enet.blocks[3][0].conv_pwl.out_channels
        self.blocks.append(My_SpatialAttention())
        self.blocks.append(self.enet.blocks[4])
        attn_ich = self.enet.blocks[4][0].conv_pwl.out_channels
        self.blocks.append(My_SpatialAttention())
        self.blocks.append(self.enet.blocks[5]) 
        attn_ich = self.enet.blocks[5][0].conv_pwl.out_channels
        self.blocks.append(My_SpatialAttention())
        self.blocks.append(self.enet.blocks[6]) 
        attn_ich = self.enet.blocks[6][0].conv_pwl.out_channels   
        self.blocks.append(My_SpatialAttention())
        
        self.conv_head =self.enet.conv_head 
        self.bn2 = self.enet.bn2
        self.act2 = self.enet.act2
        self.global_pool = self.enet.global_pool 
        attn_ich = self.conv_head.out_channels  
        self.classifier = nn.Linear(in_ch, out_dim)
        #self.classifier2 = nn.Linear(out_dim, 2) #for binary classification
        self.enet = nn.Identity()
   
    def extract(self, x):    
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        for block in self.blocks:
            x = block(x) 
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.global_pool(x)             
        return x

    def forward(self, x, x_meta=None):
        x = self.extract(x)
        x= x.squeeze(-1).squeeze(-1)  
        out1 = self.classifier(x)      
        return out1

