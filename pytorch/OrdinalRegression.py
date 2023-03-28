import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class OrdinalRegressionLayer(nn.module):
    def __init__(self):
        super(OrdinalRegressionLayer, self).__init__()

    def forward(self, x):
        # x로 받는 것은 ASPP 해서 모든 값 CONCAT해온 값daspp_feat
        N, C, H, W = x.size()
        ord_num = C // 2  # ordinal regression 할 라벨링 갯수는 채널에서 가져옴

        A = x[:, ::2, :, :]  # 짝수번째 인덱스만 가져옴
        B = x[:, 1::2, :, :]  # 홀수번째 인덱스만 가져옴
        A = A.unsqueeze(dim=1)
        B = B.unsqueeze(dim=1)

        concat_feats = torch.cat((A, B), dim=1)

        # 훈련모드일 때는
        if self.training:
            prob = F.log_softmax(concat_feats, dim=1)
            ord_prob = x.clone()
            ord_prob[:, 0::2, :, :] = prob[:, 0, :, :, :]
            ord_prob[:, 1::2, :, :] = prob[:, 1, :, :, :]
            return ord_prob

        ord_prob = F.softmax(concat_feats, dim=1)[:, 0, ::]
        ord_label = torch.sum((ord_prob > 0.5), dim=1).reshape((N, 1, H, W))
        return ord_prob, ord_label

        # 라벨에서는 log 를 쓰는게 아니라 loss에서 log를 써야하나?????


class OrdinalRegressionLoss(object):
    def __init__(self, ord_num, beta, discretization="SID"):
        self.ord_num = ord_num
        self.beta = beta
        self.discretization = discretization

    def _create_ord_labe(self, gt):
        N, H, W = gt.shape
        print("gt shap : ", gt.shape)

        ord_c0 = torch.ones(N, self.ord_num, H, W).to(gt.device)

        label = self.ord_num * torch.log(gt) / np.log(self.beta)
        label = label.long()  # 64비트 정수형으로 바꿔줌 라벨링 해주는 거임
        mask = torch.linspace(0, self.ord_num - 1, self.ord_num, requires_grad=False).view(1, self.ord_num, 1, 1).to(
            gt.device)
        mask = mask.repeat(N, 1, H, W).contiguous().long()
        ord_c0[mask] = 0

        ord_c1 = 1 - ord_c0
        ord_label = torch.cat((ord_c0, ord_c1), dim=1)
        return ord_label, mask

    def __call__(self, prob, gt):
        valid_mask = gt > 0.
        ord_label, mask = self._create_ord_labe(gt)
        print("prob shape : {}, ord label shape: {}".format(prob.shape, ord_label.shape))
        entropy = -prob * ord_label
        loss = torch.sum(entropy, dim=1)[valid_mask]
        return loss.mean()
