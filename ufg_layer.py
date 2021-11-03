import torch
import torch.nn as nn
import math


def multiScales(x, r, Lev, num_nodes):
    """
    calculate the scales of the high frequency wavelet coefficients, which will be used for wavelet shrinkage.

    :param x: all the blocks of wavelet coefficients, shape [r * Lev * num_nodes, num_hid_features] torch dense tensor
    :param r: an integer
    :param Lev: an integer
    :param num_nodes: an integer which denotes the number of nodes in the graph
    :return: scales stored in a torch dense tensor with shape [(r - 1) * Lev] for wavelet shrinkage
    """
    for block_idx in range(Lev, r * Lev):
        if block_idx == Lev:
            specEnergy_temp = torch.unsqueeze(torch.sum(x[block_idx * num_nodes:(block_idx + 1) * num_nodes, :] ** 2), dim=0)
            specEnergy = torch.unsqueeze(torch.tensor(1.0), dim=0).to(x.device)
        else:
            specEnergy = torch.cat((specEnergy,
                                    torch.unsqueeze(torch.sum(x[block_idx * num_nodes:(block_idx + 1) * num_nodes, :] ** 2), dim=0) / specEnergy_temp))

    assert specEnergy.shape[0] == (r - 1) * Lev, 'something wrong in multiScales'
    return specEnergy


def simpleLambda(x, scale, sigma=1.0):
    """
    De-noising by Soft-thresholding. Author: David L. Donoho

    :param x: one block of wavelet coefficients, shape [num_nodes, num_hid_features] torch dense tensor
    :param scale: the scale of the specific input block of wavelet coefficients, a zero-dimensional torch tensor
    :param sigma: a scalar constant, which denotes the standard deviation of the noise
    :return: thresholds stored in a torch dense tensor with shape [num_hid_features]
    """
    n, m = x.shape
    thr = (math.sqrt(2 * math.log(n)) / math.sqrt(n) * sigma) * torch.unsqueeze(scale, dim=0).repeat(m)

    return thr


def waveletShrinkage(x, thr, mode='soft'):
    """
    Perform soft or hard thresholding. The shrinkage is only applied to high frequency blocks.

    :param x: one block of wavelet coefficients, shape [num_nodes, num_hid_features] torch dense tensor
    :param thr: thresholds stored in a torch dense tensor with shape [num_hid_features]
    :param mode: 'soft' or 'hard'. Default: 'soft'
    :return: one block of wavelet coefficients after shrinkage. The shape will not be changed
    """
    assert mode in ('soft', 'hard'), 'shrinkage type is invalid'

    if mode == 'soft':
        x = torch.mul(torch.sign(x), (((torch.abs(x) - thr) + torch.abs(torch.abs(x) - thr)) / 2))
    else:
        x = torch.mul(x, (torch.abs(x) > thr))

    return x


class UFGConv_S(nn.Module):
    def __init__(self, in_features, out_features, r, Lev, num_nodes, shrinkage, sigma, bias=True):
        super(UFGConv_S, self).__init__()
        self.r = r
        self.Lev = Lev
        self.num_nodes = num_nodes
        self.shrinkage = shrinkage
        self.sigma = sigma
        self.crop_len = (Lev - 1) * num_nodes
        if torch.cuda.is_available():
            self.weight = nn.Parameter(torch.Tensor(in_features, out_features).cuda())
            self.filter = nn.Parameter(torch.Tensor(r * Lev * num_nodes, 1).cuda())
        else:
            self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
            self.filter = nn.Parameter(torch.Tensor(r * Lev * num_nodes, 1))
        if bias:
            if torch.cuda.is_available():
                self.bias = nn.Parameter(torch.Tensor(out_features).cuda())
            else:
                self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.filter, 0.9, 1.1)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, d_list):
        # d_list is a list of matrix operators (torch sparse format), row-by-row
        # x is a torch dense tensor
        x = torch.matmul(x, self.weight)

        # Fast Tight Frame Decomposition
        x = torch.sparse.mm(torch.cat(d_list, dim=0), x)
        # the output x has shape [r * Lev * num_nodes, #Features]

        # Hadamard product in spectral domain
        x = self.filter * x
        # filter has shape [r * Lev * num_nodes, 1]
        # the output x has shape [r * Lev * num_nodes, #Features]

        # calculate the scales for thresholding
        ms = multiScales(x, self.r, self.Lev, self.num_nodes)

        # perform wavelet shrinkage
        for block_idx in range(self.Lev - 1, self.r * self.Lev):
            ms_idx = 0
            if block_idx == self.Lev - 1:  # low frequency block
                x_shrink = x[block_idx * self.num_nodes:(block_idx + 1) * self.num_nodes, :]
            else:  # remaining high frequency blocks with wavelet shrinkage
                x_shrink = torch.cat((x_shrink,
                                      waveletShrinkage(x[block_idx * self.num_nodes:(block_idx + 1) * self.num_nodes, :],
                                                       simpleLambda(x[block_idx * self.num_nodes:(block_idx + 1) * self.num_nodes, :],
                                                                    ms[ms_idx], self.sigma), mode=self.shrinkage)), dim=0)
                ms_idx += 1

        # Fast Tight Frame Reconstruction
        x_shrink = torch.sparse.mm(torch.cat(d_list[self.Lev - 1:], dim=1), x_shrink)

        if self.bias is not None:
            x_shrink += self.bias

        return x_shrink


class UFGConv_R(nn.Module):
    def __init__(self, in_features, out_features, r, Lev, num_nodes, bias=True):
        super(UFGConv_R, self).__init__()
        self.Lev = Lev
        self.crop_len = (Lev - 1) * num_nodes
        if torch.cuda.is_available():
            self.weight = nn.Parameter(torch.Tensor(in_features, out_features).cuda())
            self.filter = nn.Parameter(torch.Tensor(r * Lev * num_nodes, 1).cuda())
        else:
            self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
            self.filter = nn.Parameter(torch.Tensor(r * Lev * num_nodes, 1))
        if bias:
            if torch.cuda.is_available():
                self.bias = nn.Parameter(torch.Tensor(out_features).cuda())
            else:
                self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.filter, 0.9, 1.1)
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, d_list):
        # d_list is a list of matrix operators (torch sparse format), row-by-row
        # x is a torch dense tensor
        x = torch.matmul(x, self.weight)

        # Fast Tight Frame Decomposition
        x = torch.sparse.mm(torch.cat(d_list, dim=0), x)
        # the output x has shape [r * Lev * num_nodes, #Features]

        # Hadamard product in spectral domain
        x = self.filter * x
        # filter has shape [r * Lev * num_nodes, 1]
        # the output x has shape [r * Lev * num_nodes, #Features]

        # Fast Tight Frame Reconstruction
        x = torch.sparse.mm(torch.cat(d_list[self.Lev - 1:], dim=1), x[self.crop_len:, :])

        if self.bias is not None:
            x += self.bias

        return x