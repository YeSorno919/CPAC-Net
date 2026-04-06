#!/usr/bin/env python3

import numpy as np

import torch
import torch.nn.functional as F

# from torchvision.transforms.functional import pad


def calc_pad_same(in_siz, out_siz, stride, ksize):
    """Calculate same padding width.
    Args:
    ksize: kernel size [I, J].
    Returns:
    pad_: Actual padding width.
    """
    return (out_siz - 1) * stride + ksize - in_siz


def conv2d_same(input, kernel, groups, bias=None, stride=1, padding=0, dilation=1):
    n, c, h, w = input.shape
    kout, ki_c_g, kh, kw = kernel.shape
    pw = calc_pad_same(w, w, 1, kw)
    ph = calc_pad_same(h, h, 1, kh)
    pw_l = pw // 2
    pw_r = pw - pw_l
    ph_t = ph // 2
    ph_b = ph - ph_t

    input_ = F.pad(input, (pw_l, pw_r, ph_t, ph_b))
    result = F.conv2d(
        input_,
        kernel,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
    )
    assert result.shape == input.shape
    return result


def gradient_central_diff(input, cuda):
    # https://github.com/nv-tlabs/GSCNN/issues/16
    # return input, input  # FIXME: typo?
    kernel = [[1, 0, -1]]
    kernel_t = (
        0.5 * torch.Tensor(kernel) * -1.0
    )  # pytorch implements correlation instead of conv
    if type(cuda) is int:
        if cuda != -1:
            kernel_t = kernel_t.cuda(device=cuda)
    else:
        if cuda is True:
            kernel_t = kernel_t.cuda()
    n, c, h, w = input.shape

    x = conv2d_same(
        input,
        kernel_t.unsqueeze(0).unsqueeze(0).repeat([c, 1, 1, 1]),
        c,
    )
    y = conv2d_same(
        input,
        kernel_t.t().unsqueeze(0).unsqueeze(0).repeat([c, 1, 1, 1]),
        c,
    )
    return x, y


def compute_single_sided_diferences(o_x, o_y, input):
    # n,c,h,w
    # input = input.clone()
    o_y[:, :, 0, :] = input[:, :, 1, :].clone() - input[:, :, 0, :].clone()
    o_x[:, :, :, 0] = input[:, :, :, 1].clone() - input[:, :, :, 0].clone()
    # --
    o_y[:, :, -1, :] = input[:, :, -1, :].clone() - input[:, :, -2, :].clone()
    o_x[:, :, :, -1] = input[:, :, :, -1].clone() - input[:, :, :, -2].clone()
    return o_x, o_y


def numerical_gradients_2d(input, cuda=False):
    """
    numerical gradients implementation over batches using torch group conv operator.
    the single sided differences are re-computed later.
    it matches np.gradient(image) with the difference than here output=x,y for an image while there output=y,x
    :param input: N,C,H,W
    :param cuda: whether or not use cuda
    :return: X,Y
    """
    n, c, h, w = input.shape
    assert h > 1 and w > 1
    x, y = gradient_central_diff(input, cuda)
    return x, y


def convTri(input, r, cuda=False):
    """
    Convolves an image by a 2D triangle filter (the 1D triangle filter f is
    [1:r r+1 r:-1:1]/(r+1)^2, the 2D version is simply conv2(f,f'))
    :param input:
    :param r: integer filter radius
    :param cuda: move the kernel to gpu
    :return:

    # ref: https://github.com/pdollar/toolbox/blob/master/channels/convTri.m

    """
    if r <= 1:
        raise ValueError()
    n, c, h, w = input.shape
    # return input  # FIXME: typo?
    f = list(range(1, r + 1)) + [r + 1] + list(reversed(range(1, r + 1)))
    kernel = torch.Tensor([f]) / (r + 1) ** 2
    if type(cuda) is int:
        if cuda != -1:
            kernel = kernel.cuda(device=cuda)
    else:
        if cuda is True:
            kernel = kernel.cuda()

    # padding w
    input_ = F.pad(input, (1, 1, 0, 0), mode="replicate")
    input_ = F.pad(input_, (r, r, 0, 0), mode="reflect")
    input_ = [input_[:, :, :, :r], input, input_[:, :, :, -r:]]
    input_ = torch.cat(input_, 3)
    t = input_

    # padding h
    input_ = F.pad(input_, (0, 0, 1, 1), mode="replicate")
    input_ = F.pad(input_, (0, 0, r, r), mode="reflect")
    input_ = [input_[:, :, :r, :], t, input_[:, :, -r:, :]]
    input_ = torch.cat(input_, 2)

    output = F.conv2d(
        input_,
        kernel.unsqueeze(0).unsqueeze(0).repeat([c, 1, 1, 1]),
        padding=0,
        groups=c,
    )
    output = F.conv2d(
        output,
        kernel.t().unsqueeze(0).unsqueeze(0).repeat([c, 1, 1, 1]),
        padding=0,
        groups=c,
    )
    return output


def compute_normal(E, cuda=False):
    # NOTE: is this the same normal used in 'STEAL'?

    if torch.sum(torch.isnan(E)) != 0:
        print("nans found here")
        import ipdb

        ipdb.set_trace()
    E_ = convTri(E, 4, cuda)
    Ox, Oy = numerical_gradients_2d(E_, cuda)
    Oxx, _ = numerical_gradients_2d(Ox, cuda)
    Oxy, Oyy = numerical_gradients_2d(Oy, cuda)

    aa = Oyy * torch.sign(-(Oxy + 1e-5)) / (Oxx + 1e-5)
    t = torch.atan(aa)
    O = torch.remainder(t, np.pi)

    if torch.sum(torch.isnan(O)) != 0:
        print("nans found here")
        import ipdb

        ipdb.set_trace()

    return O


def compute_normal_2(E, cuda=False):
    # output Oyy and Oxx... for what?

    if torch.sum(torch.isnan(E)) != 0:
        print("nans found here")
        import ipdb

        ipdb.set_trace()
    E_ = convTri(E, 4, cuda)
    Ox, Oy = numerical_gradients_2d(E_, cuda)
    Oxx, _ = numerical_gradients_2d(Ox, cuda)
    Oxy, Oyy = numerical_gradients_2d(Oy, cuda)

    aa = Oyy * torch.sign(-(Oxy + 1e-5)) / (Oxx + 1e-5)
    t = torch.atan(aa)
    O = torch.remainder(t, np.pi)

    if torch.sum(torch.isnan(O)) != 0:
        print("nans found here")
        import ipdb

        ipdb.set_trace()

    return O, (Oyy, Oxx)


def compute_grad_mag(E, cuda=False, eps=1e-6):

    # FIXME: what is `mag`?
    # magnitude?

    # if torch.sum(torch.isnan(E)) != 0:
    #     print("nans found here")
    #     import ipdb

    #     ipdb.set_trace()

    # FIXME: input of convTri is sometimes NaN
    E_ = convTri(E, 4, cuda)
    Ox, Oy = numerical_gradients_2d(E_, cuda)
    # mag = torch.sqrt(torch.mul(Ox, Ox) + torch.mul(Oy, Oy) + 1e-6)
    # mag = mag / mag.max()  # divide by zero error

    mag = torch.sqrt(torch.mul(Ox, Ox) + torch.mul(Oy, Oy) + eps)
    mag = torch.div(mag, mag.max() + eps)

    # if torch.sum(torch.isnan(mag)) != 0:
    #     print("nans found here")
    #     import ipdb

    #     ipdb.set_trace()

    assert torch.sum(torch.isinf(mag)) == 0, f"{mag}"

    return mag
