import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DiceLoss(nn.Module):

    def __init__(self, weights=None, epsilon=1e-6):
        super().__init__()
        self.weights = weights
        self.epsilon = epsilon
    
    def forward(self, output, target):
        target = F.one_hot(target, output.size(1))
        output = F.softmax(output, dim=1)
        inter = torch.sum(output*target, dim=(2,3))
        union = torch.sum(output+target, dim=(2,3))
        dice_coeff = 2 * (inter + self.epsilon) / (union + self.epsilon)
        if self.weights is not None:
            assert len(self.weights) == dice_coeff.size(1), \
                'Length of weight tensor must match the number of classes'
            dice_coeff *= self.weights
        return 1 - torch.mean(dice_coeff)  # mean over classes and batch


class TverskyLoss(nn.Module):

    def __init__(self, beta, epsilon=1e-6):
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon
    
    def forward(self, output, target):
        target = F.one_hot(target, output.size(1)).permute(0,3,1,2)
        output = F.softmax(output, dim=1)
        true_pos = torch.sum(output*target, axis=(2,3))
        false_neg = torch.sum(target*(1-output), axis=(2,3))
        false_pos = torch.sum(output*(1-target), axis=(2,3))
        tversky_coeff = (true_pos + self.epsilon) / (true_pos + self.beta*false_pos + 
                                                     (1-self.beta)*false_neg + self.epsilon)
        return 1 - torch.mean(tversky_coeff)


class FocalTverskyLoss(nn.Module):
    
    def __init__(self, beta, gamma, epsilon=1e-6):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.epsilon = epsilon
    
    def forward(self, output, target):
        target = F.one_hot(target, output.size(1)).permute(0,3,1,2)
        output = F.softmax(output, dim=1)
        true_pos = torch.sum(output*target, axis=(2,3))
        false_neg = torch.sum(target*(1-output), axis=(2,3))
        false_pos = torch.sum(output*(1-target), axis=(2,3))
        tversky_coeff = (true_pos + self.epsilon) / (true_pos + self.beta*false_pos + 
                                                     (1-self.beta)*false_neg + self.epsilon)
        focal_tversky = (1-tversky_coeff)**self.gamma
        return torch.mean(focal_tversky)


class FocalLoss(nn.Module):
    
    def __init__(self, alpha, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, output, target):
        num_classes = output.size(1)
        assert len(self.alpha) == num_classes, \
            'Length of weight tensor must match the number of classes'
        logp = F.cross_entropy(output, target, self.alpha)
        p = torch.exp(-logp)
        focal_loss = (1-p)**self.gamma*logp
 
        return torch.mean(focal_loss)

class LDAMLoss(nn.Module):
    
    def __init__(self, cls_num_list,device, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list.cpu()))
        #m_list = m_list * (max_m / np.max(m_list))
        m_list = (m_list * (max_m / torch.max(m_list))).to(device)
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        self.device = device
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight)
        #criterion = LDAMLoss(cls_num_list=a list of numer of samples in each class, max_m=0.5, s=30, weight=per_cls_weights)
        """
        max_m: The appropriate value for max_m depends on the specific dataset and the severity of the class imbalance. 
        You can start with a small value and gradually increase it to observe the impact on the model's performance. 
        If the model struggles with class separation or experiences underfitting, increasing max_m might help. However,
        be cautious not to set it too high, as it can cause overfitting or make the model too conservative.
        for ISIC go from 0.8 to 1.0
        s: The choice of s depends on the desired scale of the logits and the specific requirements of your problem. 
        It can be used to adjust the balance between the margin and the original logits. A larger s value amplifies 
        the impact of the logits and can be useful when dealing with highly imbalanced datasets. 
        You can experiment with different values of s to find the one that works best for your dataset and model.
        for ISIC 10 to 50
        """

class LMFLoss(nn.Module):
        def __init__(self,cls_num_list,device,weight,alpha=0.2,beta=0.2, gamma=2, max_m=0.8, s=10):
            super().__init__()
            self.focal_loss = FocalLoss(weight, gamma)
            self.ldam_loss = LDAMLoss(cls_num_list,device, max_m, weight=None, s=s)
            self.alpha= alpha
            self.beta = beta

        def forward(self, output, target):
            focal_loss_output = self.focal_loss(output, target)
            ldam_loss_output = self.ldam_loss(output, target)
            total_loss = self.alpha*focal_loss_output + self.beta*ldam_loss_output
            return total_loss   
             
class ComboLoss(nn.Module):
    
    def __init__(self, losses, weights):
        super().__init__()
        assert len(weights) == len(losses), \
            'Length of weight array must match the number of loss functions'
        self.losses = losses
        self.weights = weights
    
    def forward(self, output, target):
        return sum([loss(output, target)*wt for loss, wt in zip(self.losses, self.weights)])