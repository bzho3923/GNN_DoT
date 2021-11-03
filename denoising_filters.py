import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_thresholding(x, soft_eta, mode):
    """
    Perform row-wise soft thresholding.
    The row wise shrinkage is specific on E(k+1) updating
    The element wise shrinkage is specific on Z(k+1) updating

    :param x: one block of target matrix, shape[num_nodes, num_features]
    :param soft_eta: threshold scalar stores in a torch tensor
    :param mode: model types selection "row" or "element"
    :return: one block of matrix after shrinkage, shape[num_nodes, num_features]

    """
    assert mode in ('element', 'row'), 'shrinkage type is invalid (element or row)'
    if mode == 'row':
        row_norm = torch.linalg.norm(x, dim=1).unsqueeze(1)
        row_norm.clamp_(1e-12)
        row_thresh = F.relu(row_norm - soft_eta) / row_norm
        out = x * row_thresh
    else:
        out = F.relu(x - soft_eta) - F.relu(-x - soft_eta)

    return out


# TV regularization filter
class TVDenoisingADMM(nn.Module):
    def __init__(self, num_nodes, num_features, L, alpha=0.5, admm_iter=1):
        super(TVDenoisingADMM, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.L = L
        self.admm_iter = admm_iter
        self.alpha = alpha

    def forward(self, F, d):
        """
        Parameters
        ---------
        F : Graph signal to be smoothed, shape [Num_node, Num_features]
        d : Vector of normalized graph node degrees in shape [Num_node]

        :returns : Smoothed graph signal U

        """
        for k in range(self.admm_iter):
            d[d == 0] = 1  # avoid isolated nodes to prevent nan value in 1 / d
            Uk = torch.eye(self.num_nodes).to(F.device) - self.alpha * (1.0 / d).unsqueeze(1) * torch.tensor(self.L.todense()).to(F.device)
            Uk = torch.mm(Uk, F)

        return Uk


# Edge denoising filter
class EdgeDenoisingADMM(nn.Module):
    def __init__(self, num_nodes, num_features, rho, mu1_0, mu3_0, mu4_0, admm_iter):
        super(EdgeDenoisingADMM, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.admm_iter = admm_iter
        self.rho = rho
        self.mu1_max = 1e+6
        self.initial_mu1 = mu1_0
        self.mu1 = self.initial_mu1
        self.mu3_max = 1e+6
        self.initial_mu3 = mu3_0
        self.mu3 = self.initial_mu3
        self.mu4_max = 1e+6
        self.initial_mu4 = mu4_0
        self.mu4 = self.initial_mu4

    def forward(self, F, d, init_U=None, init_Z=None, init_E=None, init_Ones=None, init_Lambda1=None, init_Lambda3=None, init_Lambda4=None):
        """
        Parameters
        ----------
        F : Graph signal to be smoothed, shape [Num_node, Num_features].
        d : Vector of normalized graph node degrees in shape [Num_node, 1].
        init_Y : initialized matrix with zeros diag and other elements of 1 / (Num_nodes -1) in shape [Num_node, Num_node].
        init_Z : initialized matrix with elements of 1 / (Num_node - 1) in shape [Num_node, Num_node].
        init_E : Initialized zero matrix in shape [Num_node, Num_feature].
        init_Ones : Vector of ones in shape [Num_node, 1].
        init_Lambda1 : Initialized zero matrix in shape [Num_node, Num_feature].
        init_Lambda3 : Initialized zero vector in shape [Num_node, 1].

        :returns: Smoothed graph signal U

        """
        if init_U is None:
            Uk = F  # Initialized matrix Uk with F in dim [Num_nodes, Num_features]
        else:
            Uk = init_U
        if init_Z is None:
            Zk = torch.zeros(self.num_nodes, self.num_nodes).to(F.device)  # Initialized matrix Zk in dim [Num_nodes, Num_nodes]
        else:
            Zk = init_Z
        if init_E is None:
            Ek = torch.zeros(self.num_nodes, self.num_features).to(F.device)  # Initialized matrix Ek in dim [Num_nodes, Num_features]
        else:
            Ek = init_E
        if init_Ones is None:
            Ones = torch.ones(self.num_nodes, 1).to(F.device)  # Initialized vector Ones in dim [Num_nodes, 1]
        else:
            Ones = init_Ones
        if init_Lambda1 is None:
            Lambda1 = torch.zeros(self.num_nodes, self.num_features).to(F.device)  # Initialized matrix Lambda1 in dim [Num_nodes, Num_features]
        else:
            Lambda1 = init_Lambda1
        if init_Lambda3 is None:
            Lambda3 = torch.zeros(self.num_nodes, 1).to(F.device)  # Initialized vector Lambda3 in dim [Num_nodes, 1]
        else:
            Lambda3 = init_Lambda3
        if init_Lambda4 is None:
            Lambda4 = torch.zeros(self.num_nodes, self.num_nodes).to(F.device)  # Initialized matrix Lambda4 in dim [Num_nodes, Num_nodes]
        else:
            Lambda4 = init_Lambda4

        self.mu1 = self.initial_mu1
        self.mu3 = self.initial_mu3
        self.mu4 = self.initial_mu4
        I1 = torch.eye(self.num_nodes, self.num_nodes).to(F.device).detach()  # Identity matrix for Yk and Zk update in dim: [Num_nodes, Num_nodes]
        I2 = torch.eye(self.num_features + 1, self.num_features + 1).to(F.device).detach()  # Identity matrix for TildeUk update in dim: [Num_features + 1, Num_features + 1]

        for k in range(self.admm_iter):
            # Update Yk
            TildeUk = torch.cat((self.mu1 ** 0.5 * Uk, self.mu3 ** 0.5 * Ones), dim=1)  # TildeUk dim: [Num_features + 1, Num_features + 1]
            l_tmp = torch.linalg.cholesky(self.mu4 * I2 + TildeUk.t() @ TildeUk)
            inv_1 = (1 / self.mu4) * (I1 - TildeUk @ torch.cholesky_solve(TildeUk.t(), l_tmp))  # Use Cholesky to solve the Woodbury small matrix

            Yk = (self.mu1 * (Uk - Ek) @ Uk.t() + self.mu3 * Ones @ Ones.t() +
                  self.mu4 * Zk + Lambda1 @ Uk.t() - Lambda3 @ Ones.t() - Lambda4) @ inv_1

            # Update Zk
            R = soft_thresholding(Yk + 1 / self.mu4 * Lambda4, 1 / self.mu4, 'element')
            torch.diagonal(R).zero_()
            Zk = R

            # Update Uk
            L0k = I1 - Yk  # L0k dim:[Num_nodes, Num_nodes]
            inv_2 = torch.diag(d) + self.mu1 * L0k.t() @ L0k
            l_inv_2 = torch.linalg.cholesky(inv_2)  # Cholesky factorize matrix into lower-tri matrix
            Uk = torch.cholesky_solve(d.unsqueeze(1) * F + L0k.t() @
                                      (self.mu1 * Ek - Lambda1), l_inv_2)

            # Update Ek
            Ek = soft_thresholding(torch.mm(L0k, Uk) + Lambda1 / self.mu1, 1 / self.mu1, 'row')
            # Update Lambda1, Lambda2, Lambda4
            Lambda1 = Lambda1 + self.mu1 * (Uk - torch.mm(Zk, Uk) - Ek)
            Lambda3 = Lambda3 + self.mu3 * (torch.sum(Zk, 1, keepdim=True) - 1)
            Lambda4 = Lambda4 + self.mu4 * (Yk - Zk)
            # Update mu1, mu3, and mu4
            self.mu1 = min(self.rho * self.mu1, self.mu1_max)
            self.mu3 = min(self.rho * self.mu3, self.mu3_max)
            self.mu4 = min(self.rho * self.mu4, self.mu4_max)

        return Uk, Zk


# Node denoising filter
class NodeDenoisingADMM(nn.Module):
    def __init__(self, num_nodes, num_features, r, J, nu, admm_iter, rho, mu2_0):
        super(NodeDenoisingADMM, self).__init__()
        self.r = r
        self.J = J
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.admm_iter = admm_iter
        self.rho = rho
        self.nu = [nu] * J
        for i in range(J):
            self.nu[i] = self.nu[i] / np.power(4.0, i)  # from (4.3) in Dong's paper
        self.nu = [0.0] + self.nu  # To include W_{0,J}
        self.mu2_max = 1e+6
        self.initial_mu2 = mu2_0
        self.mu2 = self.initial_mu2

    def forward(self, F, W_list, d, init_Qs=None, init_Lambda2=None):
        """
        Parameters
        ----------
        F : Graph signal to be smoothed, shape [Num_node, Num_features].
        W_list : Framelet Base Operator, in list, each is a sparse matrix of size Num_node x Num_node.
        d : Vector of normalized graph node degrees in shape [Num_node, 1].
        init_Qs: Initialized list of (length: j * l) zero matrix in shape [Num_node, Num_feature].
        init_Lambda2: Initialized lists of (length: j*l) zero matrix in shape [Num_node, Num_feature].

        :returns:  Smoothed graph signal U

        """
        if init_Qs is None:
            Qs = []
            for j in range(self.r - 1):
                for l in range(self.J):
                    Qs.append(torch.zeros(torch.Size([self.num_nodes, self.num_features])).to(F.device))
            Qs = [torch.zeros((self.num_nodes, self.num_features)).to(F.device)] + Qs
        else:
            Qs = init_Qs
        if init_Lambda2 is None:
            Lambda2 = []
            for j in range(self.r - 1):
                for l in range(self.J):
                    Lambda2.append(torch.zeros(torch.Size([self.num_nodes, self.num_features])).to(F.device))
            Lambda2 = [torch.zeros((self.num_nodes, self.num_features)).to(F.device)] + Lambda2
        else:
            Lambda2 = init_Lambda2

        self.mu2 = self.initial_mu2

        for k in range(self.admm_iter):
            tmp = [self.mu2 * q_jl + Lambda_jl for q_jl, Lambda_jl in zip(Qs, Lambda2)]
            # Equation (15) in the manuscript to update U
            Uk = (1.0 / (d + self.mu2)).unsqueeze(1) * (d.unsqueeze(1) * F + torch.sparse.mm(torch.cat(W_list, dim=1), torch.cat(tmp, dim=0)))
            Qs = [soft_thresholding(torch.sparse.mm(W_jl, Uk) - Lambda_jl / self.mu2, (nu_jl / self.mu2) * d.unsqueeze(1), 'element')
                  for nu_jl, W_jl, Lambda_jl in zip(self.nu, W_list, Lambda2)]
            Lambda2 = [Lambda_jl + self.mu2 * (q_jl - torch.sparse.mm(W_jl, Uk)) for W_jl, Lambda_jl, q_jl in zip(W_list, Lambda2, Qs)]
            self.mu2 = min(self.rho * self.mu2, self.mu2_max)

        return Uk


class BregmanADMM(nn.Module):
    def __init__(self, num_nodes, num_features, r, J, nu, rho, mu1_0, mu2_0,
                 mu3_0, mu4_0, lam, admm_iter):
        super(BregmanADMM, self).__init__()
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.r = r
        self.J = J
        self.admm_iter = admm_iter
        self.rho = rho
        self.nu = [nu] * J
        for i in range(J):
            self.nu[i] = self.nu[i] / np.power(4.0, i)  # Dong bin's paper section (4.3)
        self.nu = [0.0] + self.nu
        self.mu1_max = 1e+6
        self.initial_mu1 = mu1_0
        self.mu1 = self.initial_mu1
        self.mu2_max = 1e+6
        self.initial_mu2 = mu2_0
        self.mu2 = self.initial_mu2
        self.mu3_max = 1e+6
        self.initial_mu3 = mu3_0
        self.mu3 = self.initial_mu3
        self.mu4_max = 1e+6
        self.initial_mu4 = mu4_0
        self.mu4 = self.initial_mu4
        self.lam = lam

    def forward(self, F, W_list, d, init_U=None, init_Qs=None, init_Z=None,
                init_E=None, init_Ones=None, init_Lambda1=None, init_Lambda2=None,
                init_Lambda3=None, init_Lambda4=None):
        """
        Parameters
        ----------
        init_U : Initialized matrix U with F in dim [Num_nodes, Num_features].
        F : Graph signal to be smoothed, shape [Num_node, Num_features].
        W_list : Framelet Base Operator, in list, each is a sparse matrix of size [Num_node, Num_node].
        d : Vector of normalized graph node degrees in shape [Num_node].
        init_Qs : Initialized list of (length: j * l) zero matrix in shape [Num_node, Num_feature].
        init_Z : Initialized zero matrix in shape [Num_node, Num_node].
        init_E : Initialized zero matrix in shape [Num_node, Num_feature].
        init_Ones : Vector of ones in shape [Num_node, 1].
        init_Lambda1 : Initialized zero matrix in shape [Num_node, Num_feature].
        init_Lambda2 : Initialized lists of (length: j * l) zero matrix in shape [Num_node, Num_feature].
        init_Lambda3 : Initialized zero vector in shape [Num_node, 1].
        init_Lambda4 : Initialized Zero matrix in shape [Num_node, Num_node].

        :returns:
        Uk : Smoothed graph signal U. (feature matrix)
        Zk : Smoothed graph topology structure (adjacency matrix)

        """
        if init_U is None:
            Uk = F  # Initialized matrix Uk with F in dim [Num_nodes, Num_features]
        else:
            Uk = init_U

        if init_Qs is None:
            Qs = []
            for j in range(self.r - 1):
                for l in range(self.J):
                    Qs.append(torch.zeros(torch.Size([self.num_nodes, self.num_features])).to(F.device))
            Qs = [torch.zeros((self.num_nodes, self.num_features)).to(F.device)] + Qs
        else:
            Qs = init_Qs

        if init_Z is None:
            Zk = torch.zeros(self.num_nodes, self.num_nodes).to(F.device)  # Initialized matrix Zk in dim [Num_nodes, Num_nodes]
        else:
            Zk = init_Z

        if init_E is None:
            Ek = torch.zeros(self.num_nodes, self.num_features).to(F.device)  # Initialized matrix Ek in dim [Num_nodes, Num_features]
        else:
            Ek = init_E

        if init_Ones is None:
            Ones = torch.ones(self.num_nodes, 1).to(F.device).detach()  # Initialized vector Ones in dim [Num_nodes, 1]
        else:
            Ones = init_Ones

        if init_Lambda1 is None:
            Lambda1 = torch.zeros(self.num_nodes, self.num_features).to(F.device)  # Initialized matrix Lambda1 in dim [Num_nodes, Num_features]
        else:
            Lambda1 = init_Lambda1

        if init_Lambda2 is None:
            Lambda2 = []
            for j in range(self.r - 1):
                for l in range(self.J):
                    Lambda2.append(torch.zeros(self.num_nodes, self.num_features).to(F.device))
            Lambda2 = [torch.zeros(self.num_nodes, self.num_features).to(F.device)] + Lambda2
        else:
            Lambda2 = init_Lambda2  # Setting initial Lambda2 as list (length: j * l) of zero matrix with dim[n * d]

        if init_Lambda3 is None:
            Lambda3 = torch.zeros(self.num_nodes, 1).to(F.device)  # Initialized vector Lambda3 in dim [Num_nodes, 1]
        else:
            Lambda3 = init_Lambda3

        if init_Lambda4 is None:
            Lambda4 = torch.zeros(self.num_nodes, self.num_nodes).to(F.device)  # Initialized matrix Lambda4 in dim [Num_nodes, Num_nodes]
        else:
            Lambda4 = init_Lambda4

        self.mu1 = self.initial_mu1
        self.mu2 = self.initial_mu2
        self.mu3 = self.initial_mu3
        self.mu4 = self.initial_mu4

        I1 = torch.eye(self.num_nodes, self.num_nodes).to(F.device)  # Identity matrix for Yk and Zk update in dim: [Num_nodes, Num_nodes]
        I2 = torch.eye(self.num_features + 1, self.num_features + 1).to(F.device)  # Identity matrix for TildeUk update in dim: [Num_features + 1, Num_features + 1]
        for k in range(self.admm_iter):
            # Update Yk
            TildeUk = torch.cat((self.mu1 ** 0.5 * Uk, self.mu3 ** 0.5 * Ones), dim=1)  # TildeUk dim: [Num_features + 1, Num_features + 1]
            inv_1 = (1 / self.mu4) * (I1 - TildeUk @ torch.linalg.solve(
                self.mu4 * I2 + TildeUk.t() @ TildeUk, TildeUk.t()))  # Solution of Woodbury formula scaling down the large inverse
            Yk = (self.mu1 * (Uk - Ek) @ Uk.t() + self.mu3 * Ones @ Ones.t() +
                  self.mu4 * Zk + Lambda1 @ Uk.t() - Lambda3 @ Ones.t() - Lambda4) @ inv_1
            # Update Zk
            R = soft_thresholding(Yk + 1 / self.mu4 * Lambda4, 1 / self.mu4, 'element')
            torch.diagonal(R).zero_()
            Zk = R
            # Update Uk
            L0k = I1 - Yk
            inv_2 = torch.diag(d + self.mu2) + self.mu1 * L0k.t() @ L0k
            tmp_1 = [self.mu2 * q_jl + Lambda2_jl for q_jl, Lambda2_jl in zip(Qs, Lambda2)]  # mu2 * Qjl + Lambda2jl
            tmp_2 = d.unsqueeze(1) * F + torch.mm(L0k.t(), self.mu1 * Ek - Lambda1) + \
                    torch.sparse.mm(torch.cat(W_list, dim=1), torch.cat(tmp_1, dim=0))
            Uk = torch.linalg.solve(inv_2, tmp_2)
            # Update Qs
            Qs = [soft_thresholding(torch.sparse.mm(W_jl, Uk) - Lambda_jl / self.mu2, (nu_jl / self.mu2) * d.unsqueeze(1), 'element')
                  for nu_jl, W_jl, Lambda_jl in zip(self.nu, W_list, Lambda2)]
            # Update Ek
            Ek = soft_thresholding(torch.mm(L0k, Uk) + Lambda1 / self.mu1, 1 / self.mu1, 'row')
            # Update Lambda1, Lambda2, Lambda3, Lambda4
            Lambda1 = Lambda1 + self.mu1 * (Uk - torch.mm(Zk, Uk) - Ek)
            Lambda2 = [Lambda_jl + self.mu2 * (q_jl - torch.sparse.mm(W_jl, Uk)) for W_jl, Lambda_jl, q_jl in zip(W_list, Lambda2, Qs)]
            Lambda3 = Lambda3 + self.mu3 * (torch.sum(Zk, 1, keepdim=True) - 1)
            Lambda4 = Lambda4 + self.mu4 * (Yk - Zk)
            # Update mu1, mu3, and mu4
            self.mu1 = min(self.rho * self.mu1, self.mu1_max)
            self.mu2 = min(self.rho * self.mu2, self.mu2_max)
            self.mu3 = min(self.rho * self.mu3, self.mu3_max)
            self.mu4 = min(self.rho * self.mu4, self.mu4_max)

        return Uk, Zk