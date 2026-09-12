import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F

def masked_mse_loss(student_logits, teacher_logits, mask):
    """
    student_logits: [B, C, T, H, W]
    teacher_logits: [B, C, T, H, W]
    mask:           [B, 1, T, H, W] or [B, C, T, H, W], float (0/1) or bool
    """
    assert student_logits.shape == teacher_logits.shape
    if mask.dtype == torch.bool:
        mask = mask.float()
    squared_error = (student_logits - teacher_logits) ** 2  # [B, C, T, H, W]
    masked_error = squared_error * mask                      # apply mask
    return masked_error.sum() / (mask.sum() + 1e-8)          # avoid div by zero


class CrossEntropyLoss(nn.Module):
    def __init__(self, num_class=24):
        super(CrossEntropyLoss, self).__init__()
        self.num_class = num_class

    def forward(self, x, target):
        """
        x: [B, num_class] model logits.
        target: [B] integer sample labels.
        """
        target = target.long()
        log_probs = F.log_softmax(x, dim=1)  # [B, C]
        loss = F.nll_loss(log_probs, target, reduction="mean")
        return loss

# def JsdLoss(student_logits, teacher_logits, eps=1e-8):
#     """
#     student_logits: [B, C]
#     teacher_logits: [B, C] (soft label)
#     """
#     # Student and teacher probability distributions.
#     p = F.softmax(student_logits, dim=1)
#     q = F.softmax(teacher_logits, dim=1)

#     # Mixture distribution.
#     m = 0.5 * (p + q)

#     # KL(P || M)
#     kl_pm = torch.sum(p * torch.log((p + eps) / (m + eps)), dim=1)
#     kl_qm = torch.sum(q * torch.log((q + eps) / (m + eps)), dim=1)

#     jsd = 0.5 * (kl_pm + kl_qm)
#     return jsd.mean()

def JsdLoss(student_probs, teacher_probs, eps=1e-8):
    """
    student_probs: [B, C] probability distribution (sum=1).
    teacher_probs: [B, C] probability distribution (sum=1).
    """
    p = student_probs
    q = teacher_probs

    # Mixture distribution.
    m = 0.5 * (p + q)

    # KL(P || M) and KL(Q || M).
    kl_pm = torch.sum(p * torch.log((p) / (m + eps)), dim=1)
    kl_qm = torch.sum(q * torch.log((q) / (m + eps)), dim=1)

    jsd = 0.5 * (kl_pm + kl_qm)
    return jsd.mean()

class SpreadLoss(_Loss):

    def __init__(self, m_min=0.2, m_max=0.9, num_class=24):
        super(SpreadLoss, self).__init__()
        self.m_min = m_min
        self.m_max = m_max
        self.num_class = num_class

    def forward(self, x, target):
        r=0
        target = target.long()
        # target comes in as class number like 23
        # x comes in as a length 64 vector of averages of all locations
        
        # x - BS x # CLASSES
        b, E = x.shape
        assert E == self.num_class
        
        margin = self.m_min + (self.m_max - self.m_min) * r
        # print('predictions', x[0])
        # print('target',target[0])
        # print('margin',margin)
        # print('target',target.size())

        at = torch.cuda.FloatTensor(b).fill_(0)
        for i, lb in enumerate(target):
            at[i] = x[i][lb]
            # print('an at value',x[i][lb])
        at = at.view(b, 1).repeat(1, E)
        # print('at shape',at.shape)
        # print('at',at[0])

        zeros = x.new_zeros(x.shape)
        # print('zero shape',zeros.shape)
        absloss = torch.max(.9 - (at - x), zeros)
        loss = torch.max(margin - (at - x), zeros)
        # print('loss',loss.shape)
        # print('loss',loss)
        absloss = absloss ** 2
        loss = loss ** 2
        absloss = absloss.sum() / b - .9 ** 2
        loss = loss.sum() / b - margin ** 2
        loss = loss.sum()/b

        return loss, absloss

def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    num_classes = input_logits.size()[1]
    return F.mse_loss(input_softmax, target_softmax, size_average=False) / num_classes

def softmax_kl_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)
    return F.kl_div(input_log_softmax, target_softmax, size_average=False)


def symmetric_mse_loss(input1, input2):
    """Like F.mse_loss but sends gradients to both directions
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to both input1 and input2.
    """
    assert input1.size() == input2.size()
    num_classes = input1.size()[1]
    return torch.sum((input1 - input2)**2) / num_classes

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        
        intersection = (inputs * targets).sum()   
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class IoULoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions 
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        
        IoU = (intersection + smooth)/(union + smooth)
                
        return 1 - IoU


ALPHA = 0.8
GAMMA = 2

class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=ALPHA, gamma=GAMMA, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss
        

class CapsuleLoss(nn.Module):
    def __init__(self):
        super(CapsuleLoss, self).__init__()

    def forward(self, labels, classes):
        # print('labels',labels[0])
        # print('predictions',classes[0])
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        return margin_loss

def weighted_mse_loss(input, target, weight):

    return (weight * (input - target) ** 2).mean()
