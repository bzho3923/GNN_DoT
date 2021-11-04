from scipy import sparse
from scipy.sparse.linalg import lobpcg
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.datasets import Planetoid, WikiCS
from torch_geometric.utils import get_laplacian, degree
from ufg_layer import UFGConv_S, UFGConv_R
from denoising_filters import *
from graph_attack import edge_attack, node_attack
from utils import scipy_to_torch_sparse, get_operator
from config import parser
import random
import os.path as osp
torch.set_default_dtype(torch.float64)
torch.set_default_tensor_type(torch.DoubleTensor)


class Net(nn.Module):
    def __init__(self, conv, conv_type, smoothing, filter_type, Lev, dropout_prob=0.5):
        super(Net, self).__init__()
        self.GConv = conv
        self.conv_type = conv_type
        self.smoothing = smoothing
        self.filter_type = filter_type
        self.Lev = Lev
        self.drop1 = nn.Dropout(dropout_prob)
        assert filter_type.lower() in ['tv', 'node', 'edge', 'breg', 'none'], 'invalid filter type'
        assert conv_type.lower() in ['gcn', 'gat', 'ufg_r', 'ufg_s'], 'invalid graph convolution type'
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.GConv:
            conv.reset_parameters()

    def forward(self, data, d_list, degree):
        x, edge_index = data.x, data.edge_index

        if 'ufg' in self.conv_type.lower():
            x = self.GConv[0](x, d_list)
        else:
            x = self.GConv[0](x, edge_index)

        if self.filter_type.lower() == 'none':
            if self.conv_type.lower() == 'gat':
                x = F.elu(x)
            elif self.conv_type.lower() == ('gcn' or 'ufg_r'):
                x = F.relu(x)

        x = self.drop1(x)
        
        if self.filter_type.lower() == 'tv':
            x = self.smoothing(x, degree)
            x = self.drop1(x)

        elif self.filter_type.lower() == 'node':
            x = self.smoothing(x, d_list[self.Lev - 1:], degree)
            x = self.drop1(x)
        
        elif self.filter_type.lower() == 'edge':
            x, _ = self.smoothing(x, degree)
            x = self.drop1(x)
        
        elif self.filter_type.lower() == 'breg':
            x, _ = self.smoothing(x, d_list[self.Lev - 1:], degree)
            x = self.drop1(x)

        if 'ufg' in self.conv_type.lower():
            x = self.GConv[1](x, d_list)
        else:
            x = self.GConv[1](x, edge_index)

        return F.log_softmax(x, dim=-1)


if __name__ == '__main__':
    # get config
    args = parser.parse_args()
    if args.filter_type.lower() == 'dot':
        args.filter_type = 'Breg'

    # set random seed for reproducible results
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # training on CPU/GPU device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # load dataset
    dataname = args.dataset
    rootname = osp.join(osp.abspath(''), 'data', dataname)
    if dataname.lower() == 'wikics':
        dataset = WikiCS(root=rootname)
    else:
        dataset = Planetoid(root=rootname, name=dataname)
    data = dataset[0]
    num_nodes = data.x.shape[0]
    num_features = data.x.shape[1]
    if dataname.lower() == 'wikics':
        data['train_mask'] = data['train_mask'][:, 0]
        data['val_mask'] = data['val_mask'][:, 0]

    # attack (optional)
    if args.attack.lower() == 'edge':
        edge_index_attack = edge_attack(data, 0.750, data.edge_index)  # first time edge attack by reducing 25% edge volumes
        edge_index_attack = edge_attack(data, 1.333, edge_index_attack)  # second time edge attack by increasing 25% edge volumes
        data.edge_index = edge_index_attack
    elif args.attack.lower() == 'node':
        if dataname.lower() == 'wikics':
            x_attack = node_attack(data.x, 0.50, normal=True)
        else:
            x_attack = node_attack(data.x, 0.50)
        data.x = x_attack
    elif args.attack.lower() == 'mix':
        edge_index_attack = edge_attack(data, 0.875, data.edge_index)  # first time edge attack by reducing 12.5% edge volumes
        edge_index_attack = edge_attack(data, 1.143, edge_index_attack)  # second time edge attack by increasing 12.5% edge volumes
        data.edge_index = edge_index_attack
        if dataname.lower() == 'wikics':
            x_attack = node_attack(data.x, 0.25, normal=True)  # 25% node attack
        else:
            x_attack = node_attack(data.x, 0.25)  # 25% node attack
        data.x = x_attack
    elif args.attack.lower() == 'none':
        pass
    else:
        raise Exception('invalid attack type')

    # get graph Laplacian
    L = get_laplacian(data.edge_index, num_nodes=num_nodes, normalization='sym')
    L = sparse.coo_matrix((L[1].numpy(), (L[0][0, :].numpy(), L[0][1, :].numpy())), shape=(num_nodes, num_nodes))

    # get maximum eigenvalues of the graph Laplacian
    lobpcg_init = np.random.rand(num_nodes, 1)
    lambda_max, _ = lobpcg(L, lobpcg_init)
    lambda_max = lambda_max[0]
    
    # get degrees
    deg = degree(data.edge_index[0], num_nodes).to(device)

    # extract decomposition/reconstruction Masks
    D1 = lambda x: np.cos(x / 2)
    D2 = lambda x: np.sin(x / 2)
    DFilters = [D1, D2]
    RFilters = [D1, D2]

    # get matrix operators
    J = np.log(lambda_max / np.pi) / np.log(args.s) + args.Lev - 1  # dilation level to start the decomposition
    d = get_operator(L, DFilters, args.n, args.s, J, args.Lev)
    # enhance sparseness of the matrix operators (optional)
    # d[np.abs(d) < 0.001] = 0.0

    # store the matrix operators (torch sparse format) into a list: row-by-row
    r = len(DFilters)
    d_list = list()
    for i in range(r):
        for l in range(args.Lev):
            d_list.append(scipy_to_torch_sparse(d[i, l]).to(device))

    # send data to the device
    data = data.to(device)

    # initialize graph convolution module
    if args.GConv_type.lower() == 'gcn':
        GConv = nn.ModuleList([GCNConv(num_features, args.nhid), GCNConv(args.nhid, dataset.num_classes)]).to(device)
    elif args.GConv_type.lower() == 'gat':
        GConv = nn.ModuleList([GATConv(num_features, args.nhid), GATConv(args.nhid, dataset.num_classes)]).to(device)
    elif args.GConv_type.lower() == 'ufg_s':
        GConv = nn.ModuleList([UFGConv_S(num_features, args.nhid, r, args.Lev, num_nodes, shrinkage=args.shrinkage, sigma=args.sigma),
                               UFGConv_S(args.nhid, dataset.num_classes, r, args.Lev, num_nodes, shrinkage=args.shrinkage, sigma=args.sigma)]).to(device)
    elif args.GConv_type.lower() == 'ufg_r':
        GConv = nn.ModuleList([UFGConv_R(num_features, args.nhid, r, args.Lev, num_nodes),
                               UFGConv_R(args.nhid, dataset.num_classes, r, args.Lev, num_nodes)])
    else:
        raise Exception('invalid type of graph convolution')

    # initialize the denoising filter
    if args.filter_type.lower() == 'tv':
        smoothing = TVDenoisingADMM(num_nodes, args.nhid, L)
    elif args.filter_type.lower() == 'node':
        smoothing = NodeDenoisingADMM(num_nodes, args.nhid, r, args.Lev, args.nu, args.admm_iter,
                                      args.rho, args.mu2_0)
    elif args.filter_type.lower() == 'edge':
        smoothing = EdgeDenoisingADMM(num_nodes, args.nhid, args.rho, args.mu1_0, args.mu3_0,
                                      args.mu4_0, args.admm_iter)
    elif args.filter_type.lower() == 'breg':
        smoothing = BregmanADMM(num_nodes, args.nhid, r, args.Lev, args.nu, args.rho, args.mu1_0,
                                args.mu2_0, args.mu3_0, args.mu4_0, args.lam, args.admm_iter)
    elif args.filter_type.lower() == 'none':
        smoothing = None
    else:
        raise Exception('invalid FilterType')

    # create result matrices
    num_epochs = args.epochs
    num_reps = args.reps
    epoch_loss = dict()
    epoch_acc = dict()
    epoch_loss['train_mask'] = np.zeros((num_reps, num_epochs))
    epoch_acc['train_mask'] = np.zeros((num_reps, num_epochs))
    epoch_loss['val_mask'] = np.zeros((num_reps, num_epochs))
    epoch_acc['val_mask'] = np.zeros((num_reps, num_epochs))
    epoch_loss['test_mask'] = np.zeros((num_reps, num_epochs))
    epoch_acc['test_mask'] = np.zeros((num_reps, num_epochs))
    saved_model_val_acc = np.zeros(num_reps)
    saved_model_test_acc = np.zeros(num_reps)

    # initialize the model
    model = Net(GConv, args.GConv_type, smoothing, args.filter_type, args.Lev, dropout_prob=args.dropout).to(device)

    for rep in range(num_reps):
        print('****** Rep {}: training start ******'.format(rep + 1))
        max_acc = 0.0
        
        # reset the model parameters
        model.reset_parameters()

        # initialize the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

        # training
        for epoch in range(num_epochs):
            # training mode
            model.train()
            optimizer.zero_grad()
            out = model(data, d_list, deg)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()

            # evaluation mode
            with torch.no_grad():
                model.eval()
                out = model(data, d_list, deg)
                for i, mask in data('train_mask', 'val_mask', 'test_mask'):
                    pred = out[mask].max(dim=1)[1]
                    correct = float(pred.eq(data.y[mask]).sum().item())
                    e_acc = correct / mask.sum().item()
                    epoch_acc[i][rep, epoch] = e_acc
                    e_loss = F.nll_loss(out[mask], data.y[mask])
                    epoch_loss[i][rep, epoch] = e_loss

            # print out results
            print('Epoch: {:3d}'.format(epoch + 1),
                  'train_loss: {:.4f}'.format(epoch_loss['train_mask'][rep, epoch]),
                  'train_acc: {:.4f}'.format(epoch_acc['train_mask'][rep, epoch]),
                  'val_loss: {:.4f}'.format(epoch_loss['val_mask'][rep, epoch]),
                  'val_acc: {:.4f}'.format(epoch_acc['val_mask'][rep, epoch]),
                  'test_loss: {:.4f}'.format(epoch_loss['test_mask'][rep, epoch]),
                  'test_acc: {:.4f}'.format(epoch_acc['test_mask'][rep, epoch]))

            # save model
            if epoch_acc['val_mask'][rep, epoch] > max_acc:
                torch.save(model.state_dict(), args.filename + '.pth')
                print('=== Model saved at epoch: {:3d}'.format(epoch + 1))
                max_acc = epoch_acc['val_mask'][rep, epoch]
                record_test_acc = epoch_acc['test_mask'][rep, epoch]

        saved_model_val_acc[rep] = max_acc
        saved_model_test_acc[rep] = record_test_acc
        print('#### Rep {0:2d} Finished! val acc: {1:.4f}, test acc: {2:.4f} ####\n'.format(rep + 1, max_acc, record_test_acc))

    print('***************************************************************************************************************************')
    print('Average test accuracy over {0:2d} reps: {1:.4f} with stdev {2:.4f}'.format(num_reps, np.mean(saved_model_test_acc), np.std(saved_model_test_acc)))
    print('\n')
    print(args)
    print(args.filename + '.pth', 'contains the saved model and ', args.filename + '.npz', 'contains all the values of loss and accuracy.')
    print('***************************************************************************************************************************')

    # save the results
    np.savez(args.filename + '.npz',
             epoch_train_loss=epoch_loss['train_mask'],
             epoch_train_acc=epoch_acc['train_mask'],
             epoch_valid_loss=epoch_loss['val_mask'],
             epoch_valid_acc=epoch_acc['val_mask'],
             epoch_test_loss=epoch_loss['test_mask'],
             epoch_test_acc=epoch_acc['test_mask'],
             saved_model_val_acc=saved_model_val_acc,
             saved_model_test_acc=saved_model_test_acc)