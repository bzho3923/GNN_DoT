import numpy as np
import torch
import random
from scipy.sparse import coo_matrix, tril
from torch_geometric.utils import to_dense_adj, sort_edge_index


def to_undirected_my(edge_index):
    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)
    return edge_index


# global edge attack module
def edge_attack(dataset, ratio, edgeinx):
    edge_old = edgeinx
    n = dataset.num_nodes

    m = torch.zeros((n, n))
    m[edge_old[0], edge_old[1]] = 1
    tril_idx = torch.tril_indices(n, n)

    num_nodes = dataset.num_nodes
    edge_index = edgeinx

    if ratio < 1:
        # prepare idx of the upper triangular matrix
        m[tril_idx[0], tril_idx[1]] = 0
        uptri_idx = m.to_sparse(2).indices()
        num_edge = round((1 - ratio) * len(edge_old[0]) / 2)
        # random sample
        r, c = zip(*random.sample(list(zip(uptri_idx[0], uptri_idx[1])), num_edge))
        # prepare the new adj matrix (dense)
        m[torch.stack(r), torch.stack(c)] = 0
        sym_m = m + m.t()
        edge = sym_m.to_sparse(2).indices()

    elif ratio >= 1:
        # convert to scipy sparse which is more powerful than torch sparse
        new_adj = 1 - to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
        new_adj = coo_matrix(new_adj.numpy())
        lower_tri = tril(new_adj, k=-1)
        row = torch.from_numpy(lower_tri.row)
        col = torch.from_numpy(lower_tri.col)
        new_edge_index = torch.vstack((row, col))
        num_edge_add = round((ratio - 1) * len(edge_index[0]) / 2)
        mask = np.array([0] * len(row))
        mask[:num_edge_add] = 1
        np.random.shuffle(mask)
        mask = torch.from_numpy(mask).type(torch.bool)
        add_edge = new_edge_index[:, mask]
        add_edge = to_undirected_my(add_edge)
        edge_index = torch.cat((edge_index, add_edge), dim=1)
        edge = sort_edge_index(edge_index, num_nodes=num_nodes)

    return edge


# global node attack module
def node_attack(x, ratio, normal=False):
    if normal:
        x_new = x + torch.normal(0, ratio, size=(x.shape[0], x.shape[1]))
    else:
        mask = torch.FloatTensor(x.shape[0], x.shape[1]).uniform_() < ratio
        mask = mask
        x -= mask.int()
        x_new = (torch.abs(x) == 1).double()
    return x_new