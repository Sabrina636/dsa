import torch
import torch.nn as nn
import torch.nn.functional as F



# __all__ = ['BCEDiceLoss']


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice

#
# def compute_kl_loss(p, q):
#     p_loss = F.kl_div(F.log_softmax(p, dim=-1),
#                       F.softmax(q, dim=-1), reduction='none')
#     q_loss = F.kl_div(F.log_softmax(q, dim=-1),
#                       F.softmax(p, dim=-1), reduction='none')
#
#     p_loss = p_loss.mean()
#     q_loss = q_loss.mean()
#
#     loss = (p_loss + q_loss) / 2
#     return loss
#
#
# class ClDiceLoss(nn.Module):
#     def __init__(self, alpha=0.7, beta=0.3, epsilon=1e-6):
#         super().__init__()
#         self.alpha = alpha
#         self.beta = beta
#         self.epsilon = epsilon
#
#     def forward(self, pred, target):
#         # 传统Dice Loss
#         dice_loss = 1 - self.dice_coeff(pred, target)
#
#         # 近似中心线提取（GPU加速）
#         pred_center = self.approximate_thinning(pred)
#         target_center = self.approximate_thinning(target)
#         c_dice_loss = 1 - self.dice_coeff(pred_center, target_center)
#
#         return self.alpha * dice_loss + self.beta * c_dice_loss
#
#     def dice_coeff(self, pred, target):
#         intersection = (pred * target).sum()
#         union = pred.sum() + target.sum()
#         return (2 * intersection + self.epsilon) / (union + self.epsilon)
#
#     def approximate_thinning(self, tensor):
#         binary = (tensor > 0.5).float()
#         pooled = F.max_pool2d(binary, kernel_size=3, stride=1, padding=1)
#         return binary - pooled