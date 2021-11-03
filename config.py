import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora',
                    help='name of dataset (default: Cora)')
parser.add_argument('--reps', type=int, default=10,
                    help='number of repetitions (default: 10)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.005,
                    help='learning rate (default: 5e-3)')
parser.add_argument('--wd', type=float, default=0.005,
                    help='weight decay (default: 5e-3)')
parser.add_argument('--nhid', type=int, default=16,
                    help='number of hidden units (default: 16)')
parser.add_argument('--Lev', type=int, default=2,
                    help='level of transform (default: 2)')
parser.add_argument('--s', type=float, default=2,
                    help='dilation scale > 1 (default: 2)')
parser.add_argument('--n', type=int, default=2,
                    help='n - 1 = Degree of Chebyshev Polynomial Approximation (default: n = 2)')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout probability (default: 0.5)')
parser.add_argument('--shrinkage', type=str, default='soft',
                    help='soft or hard thresholding (default: soft)')
parser.add_argument('--sigma', type=float, default=1.0,
                    help='standard deviation of the noise (default: 1.0)')
parser.add_argument('--nu', type=float, default=5.0,
                    help='tight wavelet frame transform tuning parameter (default: 5.0)')
parser.add_argument('--admm_iter', type=int, default=10,
                    help='number of admm iterations (default: 10)')
parser.add_argument('--rho', type=float, default=1.1,
                    help='piecewise function: constant and > 1 (default: 1.1)')
parser.add_argument('--mu1_0', type=float, default=9.0,
                    help='initial value of mu1 (default: 9.0)')
parser.add_argument('--mu2_0', type=float, default=3.0,
                    help='initial value of mu2 (default: 3.0)')
parser.add_argument('--mu3_0', type=float, default=1.0,
                    help='initial value of mu3 (default: 1.0)')
parser.add_argument('--mu4_0', type=float, default=1.0,
                    help='initial value of mu4 (default: 1.0)')
parser.add_argument('--lam', type=float, default=10.0,
                    help='weight of quadratic term in objective function (default: 10.0)')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 0)')
parser.add_argument('--filter_type', type=str, default='None',
                    help='denoising filter type with choices "TV", "Node", "Edge", "Breg", "None" (default: "Breg")')
parser.add_argument('--filename', type=str, default='results',
                    help='filename to store results and the model (default: results)')
parser.add_argument('--attack', type=str, default='Mix',
                    help='attack type with choices "Node", "Edge", "Mix", "None" (default: "None")')
parser.add_argument('--GConv_type', type=str, default='UFG_R',
                    help='graph convolution type with choices "GCN", "GAT", "UFG_S", "UFG_R" (default: "GCN")')