# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Custom loss functions"""
from torch import Tensor, einsum
from PolyLoss import *
import torch
from torch.nn import functional as F
from torch.autograd import Variable
import torch.nn as nn
# from util import simplex, probs2one_hot, one_hot
import torch
import gc

def clean_memory():
    """清除当前Python进程占用的GPU显存"""
    # 清除PyTorch缓存
    torch.cuda.empty_cache()
    
    # 强制垃圾回收
    gc.collect()
    
    # 再次清空缓存确保彻底
    torch.cuda.empty_cache()
    

from DualTaskLoss import DualTaskLoss

def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes

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

# def uncertainty_loss(inputs, targets):
#         """
#         Uncertainty rectified pseudo supervised loss
#         input pred and pseudo label
#         """

#         # detach from the computational graph
#         pseudo_label = F.softmax(targets / 0.5, dim=1).detach()
#         vanilla_loss = F.cross_entropy(inputs, targets, reduction='none')
#         # uncertainty rectification
#         kl_div = torch.sum(F.kl_div(F.log_softmax(inputs, dim=1), F.softmax(targets, dim=1).detach(), reduction='none'), dim=1)
#         uncertainty_loss = (torch.exp(-kl_div) * vanilla_loss).mean() + kl_div.mean()
#         return uncertainty_loss

class uncertainty_loss(nn.Module):
    def __init__(self,t=0.5):
        self.t=t
        super(uncertainty_loss,self).__init__()
        # self.poly_loss = PolyLoss(softmax=True, reduction='mean', epsilon=1)
    def forward(self,inputs, targets):
        """
        Uncertainty rectified pseudo supervised loss
        input pred and pseudo label
        """

        # detach from the computational graph
        pseudo_label = F.softmax(targets / self.t, dim=1).detach()
        vanilla_loss = F.cross_entropy(inputs, targets, reduction='none')
        # poly_loss = self.poly_loss(inputs,targets )
        # uncertainty rectification
        kl_div = torch.sum(F.kl_div(F.log_softmax(inputs, dim=1), F.softmax(targets, dim=1).detach(), reduction='none'), dim=1)
        uncertainty_loss = (torch.exp(-kl_div) * vanilla_loss).mean() + kl_div.mean()
        
        return uncertainty_loss
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
class SurfaceLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs, dist_maps) :

        # print(probs.shape,dist_maps.shape)
        pc = probs[:, self.idc, ...].type(torch.float32)
        dc = dist_maps[:, self.idc, ...].type(torch.float32)
        multipled = einsum("bkwh,bkwh->bkwh", pc, dc)

        loss = multipled.mean()

        return loss

def entropy_minmization(p):
    y1 = -1*torch.sum(p*torch.log(p+1e-6), dim=1)
    ent = torch.mean(y1)

    return ent


def entropy_map(p):
    ent_map = -1*torch.sum(p * torch.log(p + 1e-6), dim=1,
                           keepdim=True)
    return ent_map
    
class GumbelTopK(nn.Module):
    """
    Perform top-k or Gumble top-k on given data
    """
    def __init__(self, k: int, dim: int = -1, gumble: bool = False):
        """
        Args:
            k: int, number of chosen
            dim: the dimension to perform top-k
            gumble: bool, whether to introduce Gumble noise when sampling
        """
        super().__init__()
        self.k = k
        self.dim = dim
        self.gumble = gumble

    def forward(self, logits):
        # logits shape: [B, N], B denotes batch size, and N denotes the multiplication of channel and spatial dim
        if self.gumble:
            u = torch.rand(size=logits.shape, device=logits.device)
            z = - torch.log(- torch.log(u))
            return torch.topk(logits + z, self.k, dim=self.dim)
        else:
            return torch.topk(logits, self.k, dim=self.dim)


class PixelContrastiveLoss(nn.Module):
    """
    Pixel contrastive loss for segmentation. In this loss, we utilize two feature maps generated from two different
    images to calculate the contrastive loss. We treat every pixel pair belonging to the same classes as positives,
    and pixels belonging to different classes are treated as negatives.
    Since the calculation is performed on the whole image, we can sample the top-N confident pixels to calculate the
    loss and ignore other pixels.
    """
    def __init__(self, temperature: float = 0.07, sample: bool = True, sample_num: int = 400):
        """
        Args:
            temperature: hyperparameter
            sample: bool, whether to sample negative pixels
            sample_num: the number of negative samples
        """
        super().__init__()
        self.tau = temperature
        self.sample_num = sample_num
        self.sample = False if sample_num is None else sample
        self.topk = GumbelTopK(k=sample_num) if self.sample else None

    def check_input(self, input_logits, target_logits, input_seg, target_seg):
        """
        Check whether the input is valid (we expect paired inputs and only one paired inputs)
        Args:
            input_logits: input feature map class logits, shape (B, NumCls, H, W, [D])
            target_logits: target feature map class logits, shape (B, NumCls, H, W, [D])
            input_seg: input feature map class segmentation, shape (B, H, W, [D])
            target_seg: target feature map class segmentation, shape (B, H, W, [D])

        Returns:
            input_seg, target_seg, final_logits
        """
        if input_logits is not None and target_logits is not None:
            assert input_seg is None and target_seg is None, \
                'When the logits are provided, do not provide segmentation.'
            if input_logits.shape[1] == 1:  # single-class segmentation, use sigmoid activation
                input_max_probs = torch.sigmoid(input_logits)
                target_max_probs = torch.sigmoid(target_logits)
                input_cls_seg = (input_max_probs > 0.5).to(torch.float32)
                target_cls_seg = (target_max_probs > 0.5).to(torch.float32)
            else:  # multi-class segmentation case, use softmax activation
                input_max_probs, input_cls_seg = torch.max(F.softmax(input_logits, dim=1), dim=1)
                target_max_probs, target_cls_seg = torch.max(F.softmax(target_logits, dim=1), dim=1)
            return input_cls_seg, target_cls_seg, input_max_probs, target_max_probs
        elif input_seg is not None and target_seg is not None:
            assert input_logits is None and target_logits is None, \
                'When the segmentation are provided, do not provide logits.'
            confident_probs = torch.ones_like(input_seg)
            return input_seg, target_seg, confident_probs, confident_probs
        else:
            raise ValueError('The logits/segmentation are not paired, please check the input')

    def forward(self, input, target, input_logits=None, target_logits=None, input_seg=None, target_seg=None):
        """
        calculate pixel contrastive loss
        Args:
            input: input projected feature, shape (B, C, H, W, [D])
            target: target projected feature, shape (B, C, H, W, [D])
            input_logits: input feature map class logits, shape (B, NumCls, H, W, [D])
            target_logits: target feature map class logits, shape (B, NumCls, H, W, [D])
            input_seg: input feature map class segmentation, shape (B, H, W, [D])
            target_seg: target feature map class segmentation, shape (B, H, W, [D])
            logits和seg不同时出现
        Returns:
            contrastive loss
        """
        B, C, *spatial_size = input.shape  # N = H * W * D
        spatial_dims = len(spatial_size)

        # input/target shape: B * N, C
        norm_input = F.normalize(input.permute(0, *list(range(2, 2 + spatial_dims)), 1).reshape(-1, C), dim=-1)
        norm_target = F.normalize(target.permute(0, *list(range(2, 2 + spatial_dims)), 1).reshape(-1, C), dim=-1)
        pred_input, pred_target, prob_map1, prob_map2 = self.check_input(input_logits, target_logits, input_seg, target_seg)
        prob_map = (prob_map1 + prob_map2) / 2
        pred_input = pred_input.flatten()  # B * N
        pred_target = pred_target.flatten()  # B * N
        prob_map = prob_map.flatten()  # B * N

        if self.topk:
            indices = self.topk(prob_map).indices
            norm_input = norm_input[indices, :]
            norm_target = norm_target[indices, :]
            pred_input = pred_input[indices]
            pred_target = pred_target[indices]

        # mask matrix indicating whether the pixel-pair in input and target belongs to the same class or not
        diff_cls_matrix = (pred_input.unsqueeze(0) != pred_target.unsqueeze(1)).to(torch.float32)  # B * N, B * N
        # similarity matrix, which calculate the cosine distance of every pixel-pair in input and target
        # dim0 represents target and dim1 represents input
        sim_matrix = F.cosine_similarity(norm_input.unsqueeze(0), norm_target.unsqueeze(1), dim=2)
        sim_exp = torch.exp(sim_matrix / self.tau)
        nominator = torch.sum((1 - diff_cls_matrix) * sim_exp, dim=0)
        denominator = torch.sum(diff_cls_matrix * sim_exp, dim=0) + nominator
        loss = -torch.log(nominator / (denominator + 1e-8)).mean()
        return loss
    
class NegativeSamplingPixelContrastiveLoss(nn.Module):
    """
    Pixel contrastive loss for segmentation. This loss is different from the loss above, as we use two different views
    of the same image to construct the positive pairs. Negative pixels are sampled from different images according to
    the pseudo labels.
    We support bidirectional computation, which will force the negative samples to be far from both the two augmented
    views, rather than only the view chosen as anchor
    """
    def __init__(self, temperature: float = 0.07, sample_num: int = 50, bidirectional: bool = True):
        super().__init__()
        self.tau = temperature
        self.sample_num = sample_num
        self.topk = GumbelTopK(k=sample_num, dim=0)
        self.bidir = bidirectional

    def check_input(self, input_logits, target_logits, input_seg, target_seg):
        """
        Check whether the input is valid (we expect paired inputs and only one paired inputs)
        Args:
            input_logits: input feature map class logits, shape (B, NumCls, H, W, [D])
            target_logits: target feature map class logits, shape (B, NumCls, H, W, [D])
            input_seg: input feature map class segmentation, shape (B, H, W, [D])
            target_seg: target feature map class segmentation, shape (B, H, W, [D])

        Returns:
            input_seg, target_seg, final_logits
        """
        if input_logits is not None and target_logits is not None:
            assert input_seg is None and target_seg is None, \
                'When the logits are provided, do not provide segmentation.'
            if input_logits.shape[1] == 1:  # single-class segmentation, use sigmoid activation
                input_max_probs = torch.sigmoid(input_logits)
                target_max_probs = torch.sigmoid(target_logits)
                input_cls_seg = (input_max_probs > 0.5).to(torch.float32)
                target_cls_seg = (target_max_probs > 0.5).to(torch.float32)
            else:  # multi-class segmentation case, use softmax activation
                input_max_probs, input_cls_seg = torch.max(F.softmax(input_logits, dim=1), dim=1)
                target_max_probs, target_cls_seg = torch.max(F.softmax(target_logits, dim=1), dim=1)
            return input_cls_seg, target_cls_seg, input_max_probs, target_max_probs
        elif input_seg is not None and target_seg is not None:
            assert input_logits is None and target_logits is None, \
                'When the segmentation are provided, do not provide logits.'
            confident_probs = torch.ones_like(input_seg)
            return input_seg, target_seg, confident_probs, confident_probs
        else:
            raise ValueError('The logits/segmentation are not paired, please check the input')

    def forward(self, input, positive, negative, input_logits=None, negative_logits=None, input_seg=None, negative_seg=None):
        """
        calculate pixel contrastive loss
        Args:
            input: input projected feature, shape (B, C, H, W, [D])
            positive: projected feature from the other view, shape (B, C, H, W, [D])
            negative: negative projected feature, shape (B, C, H, W, [D])
            input_logits: input feature map class logits, shape (B, NumCls, H, W, [D])
            negative_logits: negative feature map class logits, shape (B, NumCls, H, W, [D])
            input_seg: input feature map class segmentation, shape (B, H, W, [D])
            negative_seg: negative feature map class segmentation, shape (B, H, W, [D])

        Returns:
            contrastive loss
        """
        B, C, *spatial_size = input.shape  # N = H * W * D
        spatial_dims = len(spatial_size)

        # input/target shape: B * N, C
        norm_input = F.normalize(input.permute(0, *list(range(2, 2 + spatial_dims)), 1).reshape(-1, C), dim=-1)
        norm_positive = F.normalize(positive.permute(0, *list(range(2, 2 + spatial_dims)), 1).reshape(-1, C), dim=-1)
        norm_negative = F.normalize(negative.permute(0, *list(range(2, 2 + spatial_dims)), 1).reshape(-1, C), dim=-1)
        seg_input, seg_negative, input_prob, negative_prob = self.check_input(
            input_logits, negative_logits, input_seg, negative_seg)

        # calculate the segmentation map to determine which pixel pairs are negative
        seg_input = seg_input.flatten()  # B * N
        seg_negative = seg_negative.flatten()  # B * N
        # input_prob = input_prob.flatten()  # B * N
        negative_prob = negative_prob.flatten()  # B * N

        # calculate the similarity between every positive pair
        positive = F.cosine_similarity(norm_input, norm_positive, dim=-1)  # B * N

        # binary matrix indicating which pixel pair is negative pair
        # seg_input is fixed in dim0, seg_target is fixed in dim1
        # in other words, dim0 represents target and dim1 represents input
        diff_cls_matrix = (seg_input.unsqueeze(0) != seg_negative.unsqueeze(1)).to(torch.float32)  # B * N, B * N
        prob_matrix = negative_prob.unsqueeze(1).expand_as(diff_cls_matrix)  # B * N, B * N
        masked_target_prob_matrix = diff_cls_matrix * prob_matrix  # mask positive pairs, sample negative only

        # sample the top-K negative pixels in the target for every pixel in the input
        sampled_negative_indices = self.topk(masked_target_prob_matrix).indices  # K, B * N
        sampled_negative = norm_negative[sampled_negative_indices]  # K, B * N, C
        negative_sim_matrix = F.cosine_similarity(norm_input.unsqueeze(0).expand_as(sampled_negative),
                                                  sampled_negative, dim=-1)  # K, B * N

        nominator = torch.exp(positive / self.tau)
        denominator = torch.exp(negative_sim_matrix / self.tau).sum(dim=0) + nominator
        loss = -torch.log(nominator / (denominator + 1e-8)).mean()
        if self.bidir:  # negative samples should be far from the other view as well
            alter_negative_sim_matrix = F.cosine_similarity(norm_positive.unsqueeze(0).expand_as(sampled_negative),
                                                            sampled_negative, dim=-1)  # K, B * N
            alter_denominator = torch.exp(alter_negative_sim_matrix / self.tau).sum(dim=0) + nominator
            alter_loss = -torch.log(nominator / (alter_denominator + 1e-8)).mean()
            loss = loss + alter_loss
        return loss
    
class BoundaryLoss(nn.Module):
    def __init__(self, beta=1):
        super(BoundaryLoss, self).__init__()
        self.beta = beta

    def forward(self, input, target):
        batch_size = input.size(0)
        loss = 0
        for i in range(batch_size):
            input_edge = self._boundary_map(input[i])
            target_edge = self._boundary_map(target[i])
            loss += F.binary_cross_entropy(input_edge, target_edge)
        return loss / batch_size

    def _boundary_map(self, x):
        sobel_x = torch.abs(F.conv2d(x.unsqueeze(0), torch.Tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]).unsqueeze(0).repeat(x.shape[1], 1, 1, 1), padding=1))
        sobel_y = torch.abs(F.conv2d(x.unsqueeze(0), torch.Tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]],[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]).unsqueeze(0).repeat(x.shape[1], 1, 1, 1), padding=1))
        edge_map = torch.sqrt(torch.pow(sobel_x, 2) + torch.pow(sobel_y, 2))
        return torch.exp(-self.beta * edge_map)
    

class ImageBasedCrossEntropyLoss2d(nn.Module):
    """
    Image Weighted Cross Entropy Loss
    """

    def __init__(
        self,
        classes,
        weight=None,
        ignore_index=255,
        norm=False,
        upper_bound=1.0,
    ):
        super(ImageBasedCrossEntropyLoss2d, self).__init__()
        # logging.info("Using Per Image based weighted loss")
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss(
            weight,
            reduction="mean",
            ignore_index=ignore_index,
        )
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = True

    def calculate_weights(self, target):
        """
        Calculate weights of classes based on the training crop
        """
        bins = torch.histc(
            target,
            bins=self.num_classes,
            min=0.0,
            max=self.num_classes,
        )
        hist_norm = bins.float() / bins.sum()
        if self.norm:
            hist = ((bins != 0).float() * self.upper_bound * (1 / hist_norm)) + 1.0
        else:
            hist = ((bins != 0).float() * self.upper_bound * (1.0 - hist_norm)) + 1.0
        return hist

    def forward(self, inputs, targets):
        if isinstance(inputs, (tuple, list)):
            inputs = inputs[0]  # get mask
        if isinstance(targets, (tuple, list)):
            targets = targets[0]  # get mask

        if self.batch_weights:
            weights = self.calculate_weights(targets)
            self.nll_loss.weight = weights

        loss = 0.0
        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                weights = self.calculate_weights(targets)
                self.nll_loss.weight = weights

            loss += self.nll_loss(
                F.log_softmax(
                    inputs[i].unsqueeze(0),
                    dim=1,
                ),
                targets[i].long().unsqueeze(0),
            )
        return loss


class JointEdgeSegLoss(nn.Module):
    def __init__(
        self,
        classes,
        ignore_index=255,
        upper_bound=1.0,
        edge_weight=1,
        seg_weight=1,
        att_weight=1,
        dual_weight=1,
    ):
        super(JointEdgeSegLoss, self).__init__()

        self.num_classes = classes
        self.seg_loss = ImageBasedCrossEntropyLoss2d(
            classes=classes,
            ignore_index=ignore_index,
            upper_bound=upper_bound,
        ).cuda()

        self.edge_weight = edge_weight
        self.seg_weight = seg_weight
        self.att_weight = att_weight
        self.dual_weight = dual_weight

        self.dual_task = DualTaskLoss(cuda=True)  # DONE: set to cuda for now

    def bce2d(self, input, target):
        n, c, h, w = input.size()

        log_p = input.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_t = target.transpose(1, 2).transpose(2, 3).contiguous().view(1, -1)
        target_trans = target_t.clone()

        pos_index = target_t == 1
        neg_index = target_t == 0
        ignore_index = target_t > 1

        target_trans[pos_index] = 1
        target_trans[neg_index] = 0

        pos_index = pos_index.data.cpu().numpy().astype(bool)
        neg_index = neg_index.data.cpu().numpy().astype(bool)
        ignore_index = ignore_index.data.cpu().numpy().astype(bool)

        weight = torch.Tensor(log_p.size()).fill_(0)
        weight = weight.numpy()
        pos_num = pos_index.sum()
        neg_num = neg_index.sum()
        sum_num = pos_num + neg_num
        weight[pos_index] = neg_num * 1.0 / sum_num
        weight[neg_index] = pos_num * 1.0 / sum_num

        weight[ignore_index] = 0

        weight = torch.from_numpy(weight)
        weight = weight.cuda()

        # done: already applies sigmoid
        # https://github.com/nv-tlabs/GSCNN/issues/62
        # https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss
        # "this loss combines a Sigmoid layer and the BCELoss in one single class."
        # done: why isn't `target_trans` used?
        loss = F.binary_cross_entropy_with_logits(
            log_p, target_t, weight, size_average=True
        )
        return loss

    def edge_attention(self, input, target, edge):
        filler = torch.ones_like(target) * 255
        return self.seg_loss(
            input,
            torch.where(edge.max(1)[0] > 0.8, target, filler),
        )

    def forward(self, segin, edgein,segmask, edgemask):


        losses = {}

        losses["seg_loss"] = self.seg_weight * self.seg_loss(segin, segmask)
        losses["edge_loss"] = self.edge_weight * 20 * self.bce2d(edgein, edgemask)

        # dual task regularizer (dual loss is left, att loss is right)
        # DONE: segin is ALL nan
        # print(segin.shape,segmask.shape)
        losses["dual_loss"] = self.dual_weight * self.dual_task(segin.float(), segmask)
        losses["att_loss"] = self.att_weight * self.edge_attention(
            segin, segmask, edgein
        )

        return losses



# 负样本不取平均
class Class_confidence_base_ContrastiveLoss_not_neg_mean(nn.Module):
    def __init__(self, threshold: float = 0.8, temperature: float = 0.07, sample_num: int = 2):
        super().__init__()
        self.tau = temperature
        self.sample_num = sample_num
        self.threshold = torch.tensor(threshold)
    def batch_contrastive_loss(self,pos_sim, anchors, negatives, batch_size=100):
        """
        分批计算对比损失
        """
        # if negatives is None or len(negatives) == 0:
        #     print("腺体锚点没有负样本")
        #     return torch.tensor(0.0, device=anchors.device)
        losses = []
        for i in range(0, negatives.size(0), batch_size):
            chunk = negatives[i:i+batch_size]
            neg_sim = F.cosine_similarity(
                anchors.unsqueeze(1),
                chunk.unsqueeze(0),
                dim=2
            )
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1) / self.tau
            losses.append(F.cross_entropy(logits, torch.zeros(logits.size(0), device=anchors.device,dtype=torch.long)))
        return torch.mean(torch.stack(losses))

    def forward(self, input, input_logits=None, input_seg=None):
        B, C, H, W = input.shape
        
        # 归一化特征 [B, H, W, C]
        norm_input = F.normalize(input.permute(0, 2, 3, 1), dim=-1)
        
        # 获取腺体和背景位置 [B, H, W]
        gland_mask = (input_seg == 1)
        bg_mask = (input_seg == 0)
        
        # 计算各类置信度
        gland_conf = input_logits[:, 1] * gland_mask.float()
        bg_conf = input_logits[:, 0] * bg_mask.float()
        
        # 计算平均置信度
        mean_gland_conf = gland_conf.sum((1,2)) / (gland_mask.sum((1,2)) + 1e-8)
        mean_bg_conf = bg_conf.sum((1,2)) / (bg_mask.sum((1,2)) + 1e-8)
        print('腺体类内平均置信度：{:2f}'.format(mean_gland_conf.detach().cpu().item()))
        print('背景类内平均置信度：{:2f}'.format(mean_bg_conf.detach().cpu().item()))
        # 获取各类样本掩码
        gland_easy_mask = (gland_conf >= mean_gland_conf.view(B,1,1)) & gland_mask
        gland_core_mask = (gland_conf >= self.threshold) & gland_mask
        gland_hard_mask = (gland_conf < mean_gland_conf.view(B,1,1)) & gland_mask
        
        bg_easy_mask = (bg_conf >= mean_bg_conf.view(B,1,1)) & bg_mask
        bg_core_mask = (bg_conf >= self.threshold) & bg_mask
        bg_hard_mask = (bg_conf < mean_bg_conf.view(B,1,1)) & bg_mask

        # 计算核心特征
        gland_core_feat = norm_input[torch.nonzero(gland_core_mask, as_tuple=True)]
        gland_core_mean = gland_core_feat.mean(dim=0, keepdim=True) if len(gland_core_feat) > 0 else None
        
        bg_core_feat = norm_input[torch.nonzero(bg_core_mask, as_tuple=True)]
        bg_core_mean = bg_core_feat.mean(dim=0, keepdim=True) if len(bg_core_feat) > 0 else None

        def sample_from_mask(mask, num_samples):
            coords = torch.nonzero(mask)
            if len(coords) == 0:
                return None
            if len(coords) > num_samples:
                idx = torch.randperm(len(coords))[:num_samples] #随机挑选num_samples个样本
                coords = coords[idx]
            return coords

        # 采样数量计算
        easy_gland_num = max(1, int(self.sample_num * (1 - mean_gland_conf.item())))
        hard_gland_num = self.sample_num - easy_gland_num
        
        easy_bg_num = max(1, int(self.sample_num * (1 - mean_bg_conf.item())))
        hard_bg_num = self.sample_num - easy_bg_num

        # 采样并修改采样得到的数量
        gland_easy_coords = sample_from_mask(gland_easy_mask, easy_gland_num)
        easy_gland_num = gland_easy_coords.shape[0]
        gland_hard_coords = sample_from_mask(gland_hard_mask, hard_gland_num)
        hard_gland_num = gland_hard_coords.shape[0]
        bg_easy_coords = sample_from_mask(bg_easy_mask, easy_bg_num)
        easy_bg_num = bg_easy_coords.shape[0]
        bg_hard_coords = sample_from_mask(bg_hard_mask, hard_bg_num)
        hard_bg_num = bg_hard_coords.shape[0]

        # 构建样本对
        anchors, positives, negatives = [], [], []
        # 表示是否进行对比学习
        gland_contractive = True
        bg_contractive = True
        # 腺体样本处理
        if gland_easy_coords is not None and gland_core_mean is not None:
            # 腺体锚点
            gland_anchors = torch.cat([gland_easy_coords, gland_hard_coords]) if gland_hard_coords is not None else gland_easy_coords
            anchors.append(gland_anchors)
            # 腺体正样本
            positives.append(gland_core_mean.expand(len(gland_anchors), -1))
            
            # 背景负样本
            bg_negatives = torch.cat([bg_easy_coords, bg_hard_coords]) if bg_hard_coords is not None else bg_easy_coords
            if bg_negatives is not None:
                negatives.append(bg_negatives)
        # 腺体核心数量为0时，此时就不进行腺体锚点的对比学习。
        elif(gland_core_mean is None):
            gland_contractive = False


        if bg_easy_coords is not None and bg_core_mean is not None:
            # 背景锚点
            bg_anchors = torch.cat([bg_easy_coords, bg_hard_coords]) if bg_hard_coords is not None else bg_easy_coords
            anchors.append(bg_anchors)
            # 背景正样本
            positives.append(bg_core_mean.expand(len(bg_anchors), -1))
            
            # 腺体负样本
            gland_negatives = torch.cat([gland_easy_coords, gland_hard_coords]) if gland_hard_coords is not None else gland_easy_coords
            if gland_negatives is not None:
                negatives.append(gland_negatives)
        # 背景核心数量为0时，此时就不进行背景锚点的对比学习。
        elif(bg_core_mean is None):
            bg_contractive = False

        # 计算损失
        if not anchors:
            return torch.tensor(0.0, device=input.device)
        #收集所有锚点（包含腺体和背景的简单/困难样本）的特征表示。
        anchors = torch.cat(anchors)
        anchor_feats = norm_input[anchors[:,0], anchors[:,1], anchors[:,2]]
        
        
        # 所有正样本特征（同类核心特征均值）拼接后的张量
        positive_feats = torch.cat(positives)
        # 计算每个锚点特征与其对应正样本特征的余弦相似度，每个锚点会与同类的核心特征均值计算相似度
        # pos_sim = F.cosine_similarity(anchor_feats, positive_feats, dim=1)
        

        if negatives:
            negatives = torch.cat(negatives)
            # if(negatives.shape[0]!=self.sample_num*2):
            #     print("负样本不足")
            # 负样本特征
            neg_feats = norm_input[negatives[:,0], negatives[:,1], negatives[:,2]]
            
            if(gland_contractive==True and bg_contractive==True): #
                anchor_gland_feats=anchor_feats[:easy_bg_num+hard_bg_num]
                positive_gland_feats = positive_feats[:easy_bg_num+hard_bg_num]
                gland_neg_feats = neg_feats[:easy_bg_num+hard_bg_num]  # 腺体锚点对应的背景负样本
                pos_gland_sim = F.cosine_similarity(anchor_gland_feats, positive_gland_feats, dim=1)
                gland_contractive_loss=self.batch_contrastive_loss(pos_gland_sim,anchor_gland_feats, gland_neg_feats)
                
                
                anchor_bg_feats=anchor_feats[easy_bg_num+hard_bg_num:]
                positive_bg_feats = positive_feats[easy_bg_num+hard_bg_num:]
                bg_neg_feats = neg_feats[easy_bg_num+hard_bg_num:]
                pos_bg_sim = F.cosine_similarity(anchor_bg_feats, positive_bg_feats, dim=1)
                bg_contractive_loss=self.batch_contrastive_loss(pos_bg_sim,anchor_bg_feats, bg_neg_feats)
                
            elif(gland_contractive==True and bg_contractive==False):#腺体进行对比，背景不进行
                anchor_gland_feats=anchor_feats
                positive_gland_feats = positive_feats
                gland_neg_feats = neg_feats  # 腺体锚点对应的背景负样本
                pos_gland_sim = F.cosine_similarity(anchor_gland_feats, positive_gland_feats, dim=1)
                gland_contractive_loss=self.batch_contrastive_loss(pos_gland_sim,anchor_gland_feats, gland_neg_feats)
                bg_contractive_loss=0
            elif(gland_contractive==False and bg_contractive==True):#腺体进行对比，背景不进行
                gland_contractive_loss=0
                
                anchor_bg_feats=anchor_feats
                positive_bg_feats = positive_feats
                bg_neg_feats = neg_feats  # 腺体锚点对应的背景负样本
                pos_bg_sim = F.cosine_similarity(anchor_bg_feats, positive_bg_feats, dim=1)
                bg_contractive_loss=self.batch_contrastive_loss(pos_bg_sim,anchor_bg_feats, bg_neg_feats)
            else:
                gland_contractive_loss=0
                bg_contractive_loss=0

        # print(gland_contractive_loss+bg_contractive_loss)
        return gland_contractive_loss+bg_contractive_loss
        
        
# 有roi取出睑板腺区域
class Class_confidence_base_ContrastiveLoss_not_neg_mean_with_roi(nn.Module):
    def __init__(self, threshold: float = 0.8, temperature: float = 0.07, sample_num: int = 2):
        super().__init__()
        self.tau = temperature
        self.sample_num = sample_num
        self.threshold = torch.tensor(threshold)
    def batch_contrastive_loss(self,pos_sim, anchors, negatives,batch_size=100):
        """
        分批计算对比损失
        """
        # if negatives is None or len(negatives) == 0:
        #     print("腺体锚点没有负样本")
        #     return torch.tensor(0.0, device=anchors.device)
        losses = []
        for i in range(0, negatives.size(0), batch_size):
            chunk = negatives[i:i+batch_size]
            neg_sim = F.cosine_similarity(
                anchors.unsqueeze(1),
                chunk.unsqueeze(0),
                dim=2
            )
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1) / self.tau
            losses.append(F.cross_entropy(logits, torch.zeros(logits.size(0), device=anchors.device,dtype=torch.long)))
        return torch.mean(torch.stack(losses))

    def forward(self, input, input_logits=None, input_seg=None,roi=None):
        B, C, H, W = input.shape
        
        # 归一化特征 [B, H, W, C]
        norm_input = F.normalize(input.permute(0, 2, 3, 1), dim=-1)
        
        # 获取腺体和背景位置 [B, H, W]
        gland_mask = (input_seg == 1)
        bg_mask = (input_seg == 0)&(roi==1)

        # 计算各类置信度
        gland_conf = input_logits[:, 1] * gland_mask.float()
        bg_conf = input_logits[:, 0] * bg_mask.float()
        
        
        # 计算平均置信度
        mean_gland_conf = gland_conf.sum((1,2)) / (gland_mask.sum((1,2)) + 1e-8)
        mean_bg_conf = bg_conf.sum((1,2)) / (bg_mask.sum((1,2)) + 1e-8)
        print('腺体类内平均置信度：{:2f}'.format(mean_gland_conf.detach().cpu().item()))
        print('背景类内平均置信度：{:2f}'.format(mean_bg_conf.detach().cpu().item()))
        # 获取各类样本掩码
        gland_easy_mask = (gland_conf >= mean_gland_conf.view(B,1,1)) & gland_mask
        gland_core_mask = (gland_conf >= self.threshold) & gland_mask
        gland_hard_mask = (gland_conf < mean_gland_conf.view(B,1,1)) & gland_mask
        
        bg_easy_mask = (bg_conf >= mean_bg_conf.view(B,1,1)) & bg_mask
        bg_core_mask = (bg_conf >= self.threshold) & bg_mask
        bg_hard_mask = (bg_conf < mean_bg_conf.view(B,1,1)) & bg_mask

        # 计算核心特征
        gland_core_feat = norm_input[torch.nonzero(gland_core_mask, as_tuple=True)]
        gland_core_mean = gland_core_feat.mean(dim=0, keepdim=True) if len(gland_core_feat) > 0 else None
        
        bg_core_feat = norm_input[torch.nonzero(bg_core_mask, as_tuple=True)]
        bg_core_mean = bg_core_feat.mean(dim=0, keepdim=True) if len(bg_core_feat) > 0 else None

        def sample_from_mask(mask, num_samples):
            coords = torch.nonzero(mask)
            if len(coords) == 0:
                return None
            if len(coords) > num_samples:
                idx = torch.randperm(len(coords))[:num_samples] #随机挑选num_samples个样本
                coords = coords[idx]
            return coords

        # 采样数量计算
        easy_gland_num = max(1, int(self.sample_num * (1 - mean_gland_conf.item())))
        hard_gland_num = self.sample_num - easy_gland_num
        
        easy_bg_num = max(1, int(self.sample_num * (1 - mean_bg_conf.item())))
        hard_bg_num = self.sample_num - easy_bg_num

        # 采样并修改采样得到的数量
        gland_easy_coords = sample_from_mask(gland_easy_mask, easy_gland_num)
        easy_gland_num = gland_easy_coords.shape[0]
        gland_hard_coords = sample_from_mask(gland_hard_mask, hard_gland_num)
        hard_gland_num = gland_hard_coords.shape[0]
        bg_easy_coords = sample_from_mask(bg_easy_mask, easy_bg_num)
        easy_bg_num = bg_easy_coords.shape[0]
        bg_hard_coords = sample_from_mask(bg_hard_mask, hard_bg_num)
        hard_bg_num = bg_hard_coords.shape[0]

        # 构建样本对
        anchors, positives, negatives = [], [], []
        # 表示是否进行对比学习
        gland_contractive = True
        bg_contractive = True
        # 腺体样本处理
        if gland_easy_coords is not None and gland_core_mean is not None:
            # 腺体锚点
            gland_anchors = torch.cat([gland_easy_coords, gland_hard_coords]) if gland_hard_coords is not None else gland_easy_coords
            anchors.append(gland_anchors)
            # 腺体正样本
            positives.append(gland_core_mean.expand(len(gland_anchors), -1))
            
            # 背景负样本
            bg_negatives = torch.cat([bg_easy_coords, bg_hard_coords]) if bg_hard_coords is not None else bg_easy_coords
            if bg_negatives is not None:
                negatives.append(bg_negatives)
        # 腺体核心数量为0时，此时就不进行腺体锚点的对比学习。
        elif(gland_core_mean is None):
            gland_contractive = False


        if bg_easy_coords is not None and bg_core_mean is not None:
            # 背景锚点
            bg_anchors = torch.cat([bg_easy_coords, bg_hard_coords]) if bg_hard_coords is not None else bg_easy_coords
            anchors.append(bg_anchors)
            # 背景正样本
            positives.append(bg_core_mean.expand(len(bg_anchors), -1))
            
            # 腺体负样本
            gland_negatives = torch.cat([gland_easy_coords, gland_hard_coords]) if gland_hard_coords is not None else gland_easy_coords
            if gland_negatives is not None:
                negatives.append(gland_negatives)
        # 背景核心数量为0时，此时就不进行背景锚点的对比学习。
        elif(bg_core_mean is None):
            bg_contractive = False

        # 计算损失
        if not anchors:
            return torch.tensor(0.0, device=input.device)
        #收集所有锚点（包含腺体和背景的简单/困难样本）的特征表示。
        anchors = torch.cat(anchors)
        anchor_feats = norm_input[anchors[:,0], anchors[:,1], anchors[:,2]]
        
        
        # 所有正样本特征（同类核心特征均值）拼接后的张量
        positive_feats = torch.cat(positives)
        # 计算每个锚点特征与其对应正样本特征的余弦相似度，每个锚点会与同类的核心特征均值计算相似度
        # pos_sim = F.cosine_similarity(anchor_feats, positive_feats, dim=1)
        

        if negatives:
            negatives = torch.cat(negatives)
            # if(negatives.shape[0]!=self.sample_num*2):
            #     print("负样本不足")
            # 负样本特征
            neg_feats = norm_input[negatives[:,0], negatives[:,1], negatives[:,2]]
            
            if(gland_contractive==True and bg_contractive==True): #
                anchor_gland_feats=anchor_feats[:easy_bg_num+hard_bg_num]
                positive_gland_feats = positive_feats[:easy_bg_num+hard_bg_num]
                gland_neg_feats = neg_feats[:easy_bg_num+hard_bg_num]  # 腺体锚点对应的背景负样本
                pos_gland_sim = F.cosine_similarity(anchor_gland_feats, positive_gland_feats, dim=1)
                gland_contractive_loss=self.batch_contrastive_loss(pos_gland_sim,anchor_gland_feats, gland_neg_feats)
                
                
                anchor_bg_feats=anchor_feats[easy_bg_num+hard_bg_num:]
                positive_bg_feats = positive_feats[easy_bg_num+hard_bg_num:]
                bg_neg_feats = neg_feats[easy_bg_num+hard_bg_num:]
                pos_bg_sim = F.cosine_similarity(anchor_bg_feats, positive_bg_feats, dim=1)
                bg_contractive_loss=self.batch_contrastive_loss(pos_bg_sim,anchor_bg_feats, bg_neg_feats)
                
            elif(gland_contractive==True and bg_contractive==False):#腺体进行对比，背景不进行
                anchor_gland_feats=anchor_feats
                positive_gland_feats = positive_feats
                gland_neg_feats = neg_feats  # 腺体锚点对应的背景负样本
                pos_gland_sim = F.cosine_similarity(anchor_gland_feats, positive_gland_feats, dim=1)
                gland_contractive_loss=self.batch_contrastive_loss(pos_gland_sim,anchor_gland_feats, gland_neg_feats)
                bg_contractive_loss=0
            elif(gland_contractive==False and bg_contractive==True):#腺体进行对比，背景不进行
                gland_contractive_loss=0
                
                anchor_bg_feats=anchor_feats
                positive_bg_feats = positive_feats
                bg_neg_feats = neg_feats  # 腺体锚点对应的背景负样本
                pos_bg_sim = F.cosine_similarity(anchor_bg_feats, positive_bg_feats, dim=1)
                bg_contractive_loss=self.batch_contrastive_loss(pos_bg_sim,anchor_bg_feats, bg_neg_feats)
            else:
                gland_contractive_loss=0
                bg_contractive_loss=0

        # print(gland_contractive_loss+bg_contractive_loss)
        return gland_contractive_loss+bg_contractive_loss
# 有roi取出睑板腺区域,没有核心像素
class Class_confidence_base_ContrastiveLoss_not_neg_mean_with_roi_wo_core(nn.Module):
    def __init__(self, temperature: float = 0.07, sample_num: int = 500):
        super().__init__()
        self.tau = temperature
        self.sample_num = sample_num
    def batch_contrastive_loss(self,pos_sim, anchors, negatives,batch_size=100):
        """
        分批计算对比损失
        """
        # if negatives is None or len(negatives) == 0:
        #     print("腺体锚点没有负样本")
        #     return torch.tensor(0.0, device=anchors.device)
        losses = []
        for i in range(0, negatives.size(0), batch_size):
            chunk = negatives[i:i+batch_size]
            neg_sim = F.cosine_similarity(
                anchors.unsqueeze(1),
                chunk.unsqueeze(0),
                dim=2
            )
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1) / self.tau
            losses.append(F.cross_entropy(logits, torch.zeros(logits.size(0), device=anchors.device,dtype=torch.long)))
        return torch.mean(torch.stack(losses))

    def forward(self, input, input_logits=None, input_seg=None,roi=None):
        B, C, H, W = input.shape
        
        # 归一化特征 [B, H, W, C]
        norm_input = F.normalize(input.permute(0, 2, 3, 1), dim=-1)
        
        # 获取腺体和背景位置 [B, H, W]
        gland_mask = (input_seg == 1)
        bg_mask = (input_seg == 0)&(roi==1)

        # 计算各类置信度
        gland_conf = input_logits[:, 1] * gland_mask.float()
        bg_conf = input_logits[:, 0] * bg_mask.float()
        
        
        # 计算平均置信度
        mean_gland_conf = gland_conf.sum((1,2)) / (gland_mask.sum((1,2)) + 1e-8)
        mean_bg_conf = bg_conf.sum((1,2)) / (bg_mask.sum((1,2)) + 1e-8)
        print('腺体类内平均置信度：{:2f}'.format(mean_gland_conf.detach().cpu().item()))
        print('背景类内平均置信度：{:2f}'.format(mean_bg_conf.detach().cpu().item()))
        # 获取各类样本掩码
        gland_easy_mask = (gland_conf >= mean_gland_conf.view(B,1,1)) & gland_mask

        gland_hard_mask = (gland_conf < mean_gland_conf.view(B,1,1)) & gland_mask
        
        bg_easy_mask = (bg_conf >= mean_bg_conf.view(B,1,1)) & bg_mask

        bg_hard_mask = (bg_conf < mean_bg_conf.view(B,1,1)) & bg_mask



        def sample_from_mask(mask, num_samples):
            coords = torch.nonzero(mask)
            if len(coords) == 0:
                return None
            if len(coords) > num_samples:
                idx = torch.randperm(len(coords))[:num_samples] #随机挑选num_samples个样本
                coords = coords[idx]
            return coords

        # 采样数量计算
        easy_gland_num = max(1, int(self.sample_num * (1 - mean_gland_conf.item())))
        hard_gland_num = self.sample_num - easy_gland_num
        
        easy_bg_num = max(1, int(self.sample_num * (1 - mean_bg_conf.item())))
        hard_bg_num = self.sample_num - easy_bg_num

        # 采样并修改采样得到的数量
        gland_easy_coords = sample_from_mask(gland_easy_mask, easy_gland_num)
        easy_gland_num = gland_easy_coords.shape[0]
        gland_hard_coords = sample_from_mask(gland_hard_mask, hard_gland_num)
        hard_gland_num = gland_hard_coords.shape[0]
        bg_easy_coords = sample_from_mask(bg_easy_mask, easy_bg_num)
        easy_bg_num = bg_easy_coords.shape[0]
        bg_hard_coords = sample_from_mask(bg_hard_mask, hard_bg_num)
        hard_bg_num = bg_hard_coords.shape[0]

        # 构建样本对
        anchors, positives, negatives = [], [], []
        # 表示是否进行对比学习
        gland_contractive = True
        bg_contractive = True
        # 腺体样本处理
        if gland_easy_coords is not None:
            # 腺体锚点
            gland_anchors = torch.cat([gland_easy_coords, gland_hard_coords]) if gland_hard_coords is not None else gland_easy_coords
            anchors.append(gland_anchors)
            # 腺体正样本,从锚点中随机选择正样本。
            shuffle_indices = (torch.randperm(gland_anchors.size(0))).to(device=input.device)
            # 根据随机索引选取正样本
            positives.append(gland_anchors[shuffle_indices])
            
            # 背景负样本
            bg_negatives = torch.cat([bg_easy_coords, bg_hard_coords]) if bg_hard_coords is not None else bg_easy_coords
            if bg_negatives is not None:
                negatives.append(bg_negatives)
        # 腺体核心数量为0时，此时就不进行腺体锚点的对比学习。
        elif(gland_easy_coords is None):
            gland_contractive = False


        if bg_easy_coords is not None:
            # 背景锚点
            bg_anchors = torch.cat([bg_easy_coords, bg_hard_coords]) if bg_hard_coords is not None else bg_easy_coords
            anchors.append(bg_anchors)
            # 背景正样本,
            shuffle_indices = (torch.randperm(bg_anchors.size(0))).to(device=input.device)
            # 根据随机索引选取正样本
            positives.append(bg_anchors[shuffle_indices])
            
            # 腺体负样本
            gland_negatives = torch.cat([gland_easy_coords, gland_hard_coords]) if gland_hard_coords is not None else gland_easy_coords
            if gland_negatives is not None:
                negatives.append(gland_negatives)
        # 背景核心数量为0时，此时就不进行背景锚点的对比学习。
        elif(bg_easy_coords is None):
            bg_contractive = False

        # 计算损失
        if not anchors:
            return torch.tensor(0.0, device=input.device)
        #收集所有锚点（包含腺体和背景的简单/困难样本）的特征表示。
        anchors = torch.cat(anchors)
        anchor_feats = norm_input[anchors[:,0], anchors[:,1], anchors[:,2]]
        
        
        # 所有正样本特征（同类核心特征均值）拼接后的张量
        positives = torch.cat(positives)
        positive_feats = norm_input[positives[:,0], positives[:,1], positives[:,2]]
 
        # 计算每个锚点特征与其对应正样本特征的余弦相似度，每个锚点会与同类的核心特征均值计算相似度
        # pos_sim = F.cosine_similarity(anchor_feats, positive_feats, dim=1)
        

        if negatives:
            negatives = torch.cat(negatives)
            # if(negatives.shape[0]!=self.sample_num*2):
            #     print("负样本不足")
            # 负样本特征
            neg_feats = norm_input[negatives[:,0], negatives[:,1], negatives[:,2]]
            
            if(gland_contractive==True and bg_contractive==True): #
                anchor_gland_feats=anchor_feats[:easy_bg_num+hard_bg_num]
                positive_gland_feats = positive_feats[:easy_bg_num+hard_bg_num]
                gland_neg_feats = neg_feats[:easy_bg_num+hard_bg_num]  # 腺体锚点对应的背景负样本
                pos_gland_sim = F.cosine_similarity(anchor_gland_feats, positive_gland_feats, dim=1)
                gland_contractive_loss=self.batch_contrastive_loss(pos_gland_sim,anchor_gland_feats, gland_neg_feats)
                
                
                anchor_bg_feats=anchor_feats[easy_bg_num+hard_bg_num:]
                positive_bg_feats = positive_feats[easy_bg_num+hard_bg_num:]
                bg_neg_feats = neg_feats[easy_bg_num+hard_bg_num:]
                pos_bg_sim = F.cosine_similarity(anchor_bg_feats, positive_bg_feats, dim=1)
                bg_contractive_loss=self.batch_contrastive_loss(pos_bg_sim,anchor_bg_feats, bg_neg_feats)
                
            elif(gland_contractive==True and bg_contractive==False):#腺体进行对比，背景不进行
                anchor_gland_feats=anchor_feats
                positive_gland_feats = positive_feats
                gland_neg_feats = neg_feats  # 腺体锚点对应的背景负样本
                pos_gland_sim = F.cosine_similarity(anchor_gland_feats, positive_gland_feats, dim=1)
                gland_contractive_loss=self.batch_contrastive_loss(pos_gland_sim,anchor_gland_feats, gland_neg_feats)
                bg_contractive_loss=0
            elif(gland_contractive==False and bg_contractive==True):#腺体进行对比，背景不进行
                gland_contractive_loss=0
                
                anchor_bg_feats=anchor_feats
                positive_bg_feats = positive_feats
                bg_neg_feats = neg_feats  # 腺体锚点对应的背景负样本
                pos_bg_sim = F.cosine_similarity(anchor_bg_feats, positive_bg_feats, dim=1)
                bg_contractive_loss=self.batch_contrastive_loss(pos_bg_sim,anchor_bg_feats, bg_neg_feats)
            else:
                gland_contractive_loss=0
                bg_contractive_loss=0

        # print(gland_contractive_loss+bg_contractive_loss)
        return gland_contractive_loss+bg_contractive_loss
if __name__ == "__main__":
    # 设置输入数据的形状
    B, C, H, W = 1, 32, 2, 3
    num_classes = 2

    # 生成随机输入数据
    input = torch.randn(B, C, H, W)
    #[1,2,2,3]
    input_logits = torch.tensor([[[[0.9,0.1,0],
                                  [0.6,0.5,0.3]],
                                 [[0.1,0.9,1],
                                  [0.4,0.5,0.7]]]])
    # [1,2,3]
    input_seg = torch.tensor([[[0,1,1],
                              [1,1,0]]])

    # 创建损失函数实例
    loss_fn = Class_confidence_base_ContrastiveLoss()

    # 计算损失
    loss = loss_fn(input, input_logits, input_seg)
    print(f"计算得到的损失值: {loss.item()}")