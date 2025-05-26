from torch import nn
import math
import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt

# ========================
# Mean-Variance Loss
# ========================
class MeanVarianceLoss(nn.Module):
    def __init__(self, lambda_1, lambda_2, start_age, end_age):
        super().__init__()
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.start_age = start_age
        self.end_age = end_age

    def forward(self, input, target):
        N = input.size()[0]
        target = target.type(torch.FloatTensor).cuda()
        m = nn.Softmax(dim=1)
        p = m(input)
        a = torch.arange(self.start_age, self.end_age + 1, dtype=torch.float32).cuda()
        mean = torch.squeeze((p * a).sum(1, keepdim=True), dim=1)
        mse = (mean - target) ** 2
        mean_loss = mse.mean() / 2.0

        b = (a[None, :] - mean[:, None]) ** 2
        variance_loss = (p * b).sum(1, keepdim=True).mean()

        return self.lambda_1 * mean_loss, self.lambda_2 * variance_loss

# ========================
# Similarity Loss
# ========================
class SimLoss(nn.Module):
    def __init__(self, number_of_classes, reduction_factor, device="cpu", epsilon=1e-8):
        super().__init__()
        self.__number_of_classes = number_of_classes
        self.__device = device
        self.epsilon = epsilon
        self.r = reduction_factor

    def forward(self, x, y):
        w = self.__w[y, :]
        return torch.mean(-torch.log(torch.sum(w * x, dim=1) + self.epsilon))

    @property
    def r(self):
        return self.__r

    @r.setter
    def r(self, r):
        assert 0.0 <= r < 1.0
        self.__r = r
        self.__w = self.__generate_w(self.__number_of_classes, self.__r, self.__device)

    def __generate_w(self, number_of_classes, reduction_factor, device):
        w = torch.zeros((number_of_classes, number_of_classes)).to(device)
        for j in range(number_of_classes):
            for i in range(number_of_classes):
                w[j, i] = reduction_factor ** np.abs(i - j)
        return w

# ========================
# Age Loss with Hinge
# ========================
HINGE_LOSS_ENABLE = True
HINGE_LAMBDA = 0.2
DELTA = 2

class loss_func(nn.Module):
    def __init__(self, device):
        super(loss_func, self).__init__()
        self.device = device
        self.cross_entropy_layer = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        softmax_loss = self.cross_entropy_layer(logits, targets)
        age_prob = logits

        if HINGE_LOSS_ENABLE:
            sign_martix = torch.ones([age_prob.shape[0], age_prob.shape[1]]).to(self.device)
            range_martix = torch.arange(0, 101).expand([age_prob.shape[0], 101]).to(self.device)
            targets_expand = targets.expand([101, age_prob.shape[0]]).t()
            sign_martix[range_martix >= targets_expand] = -1

            age_prob_part = age_prob[:, 1:]
            age_prob_move = torch.cat([age_prob_part, age_prob[:, -1].unsqueeze(1)], 1)
            hinge = (age_prob - age_prob_move) * sign_martix
            hinge = hinge + DELTA
            zero_data = torch.zeros_like(hinge)
            hinge = torch.max(hinge, zero_data)
            hinge_loss = hinge.sum(1).mean()

        result = (softmax_loss + hinge_loss * HINGE_LAMBDA, softmax_loss, hinge_loss * HINGE_LAMBDA)
        return result

# ========================
# Simple MSE Age Loss
# ========================
class AgeOnlyLoss(nn.Module):
    def __init__(self, task='regression'):
        super().__init__()
        self.task = task
        if self.task == 'regression':
            self.loss_fn = nn.MSELoss()
        elif self.task == 'group_classification':
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError("task must be 'regression' or 'group_classification'")

    def forward(self, pred, target):
        if self.task == 'regression':
            return self.loss_fn(pred.squeeze(), target.float())
        else:
            group_labels = torch.tensor(
                [self.get_group_label(a.item()) for a in target], device=target.device
            )
            return self.loss_fn(pred, group_labels)

    @staticmethod
    def get_group_label(age):
        if age < 19: return 0
        elif age < 30: return 1
        elif age < 40: return 2
        elif age < 60: return 3
        else: return 4


# ========================
# Joint Age-Gender Loss (with weighting or focal loss)
# ========================
class AgeGenderLoss(nn.Module):
    def __init__(self, age_task='regression', age_weight=1.0, gender_weight=8.0):
        super().__init__()
        self.age_task = age_task
        self.age_weight = age_weight
        self.gender_weight = gender_weight

        if self.age_task == 'regression':
            self.age_loss_fn = nn.MSELoss()
        elif self.age_task == 'group_classification':
            self.age_loss_fn = nn.CrossEntropyLoss()
        else:
            raise ValueError("age_task must be 'regression' or 'group_classification'")

        self.gender_loss_fn = FocalLoss(gamma=2.0, alpha=[0.2, 0.8])

    def forward(self, age_pred, gender_pred, age_true, gender_true):
        if self.age_task == 'regression':
            age_loss = self.age_loss_fn(age_pred.squeeze(), age_true.float())
        else:
            # ✅ 使用向量化方式计算 age_true_group，避免 for 循环 + .item() 低效写法
            age_true_group = self.get_group_label_tensor(age_true)

            age_loss = self.age_loss_fn(age_pred, age_true_group)

        gender_loss = self.gender_loss_fn(gender_pred, gender_true.long())
        return self.age_weight * age_loss + self.gender_weight * gender_loss

    @staticmethod
    def get_group_label_tensor(ages):
        group = torch.zeros_like(ages, dtype=torch.long)
        group[(ages >= 19) & (ages < 30)] = 1
        group[(ages >= 30) & (ages < 40)] = 2
        group[(ages >= 40) & (ages < 60)] = 3
        group[(ages >= 60)] = 4
        return group




class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none')
        pt = torch.exp(-ce_loss)

        if self.alpha is not None:
            alpha = self.alpha.to(input.device)
            at = alpha.gather(0, target.data.view(-1))
            ce_loss = at * ce_loss

        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class AgeGenderCascadeLoss(nn.Module):
    def __init__(self, age_weight=1.0, gender_weight=1.0):
        super().__init__()
        self.age_weight = age_weight
        self.gender_weight = gender_weight
        self.gender_loss_fn = FocalLoss(gamma=2.0, alpha=[0.2, 0.8])  # 根据性别不平衡设置
        self.age_loss_fn = nn.MSELoss()

    def forward(self, age_pred_male, age_pred_female, gender_logits, age_true, gender_true):
        # 性别分类损失
        gender_loss = self.gender_loss_fn(gender_logits, gender_true)

        # 自动根据性别标签选择对应分支的预测年龄值
        selected_age_pred = torch.where(
            gender_true.unsqueeze(1) == 0,  # 男为0
            age_pred_male,
            age_pred_female
        ).squeeze()

        # 年龄回归损失
        age_loss = self.age_loss_fn(selected_age_pred, age_true.float())

        # 总损失加权组合
        total_loss = self.age_weight * age_loss + self.gender_weight * gender_loss
        return total_loss


