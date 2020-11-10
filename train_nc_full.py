import time
import argparse
import numpy as np
import torch
from deeprobust.graph.defense import RGCN, GCN, ProGNN
from deeprobust.graph.data import Dataset, PrePtbDataset
from deeprobust.graph.utils import preprocess
from models.gin_mean_pool import GNN

import scipy.sparse as sp
import scipy
import sys

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true',
        default=False, help='debug mode')
parser.add_argument('--only_gcn', action='store_true',
        default=False, help='test the performance of gcn without other components')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=1, help='Random seed.')
parser.add_argument('--secondary_seed', type=int, default=1, help='Random seed.')

parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora',
        choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed'], help='dataset')
parser.add_argument('--attack', type=str, default='meta',
        choices=['no', 'meta', 'random', 'nettack'])
parser.add_argument('--ptb_rate', type=float, default=0.05, help="noise ptb_rate")
parser.add_argument('--epochs', type=int,  default=400, help='Number of epochs to train.')
parser.add_argument('--alpha', type=float, default=5e-4, help='weight of l1 norm')
parser.add_argument('--beta', type=float, default=1.5, help='weight of nuclear norm')
parser.add_argument('--gamma', type=float, default=1, help='weight of l2 norm')
parser.add_argument('--lambda_', type=float, default=0, help='weight of feature smoothing')
parser.add_argument('--phi', type=float, default=0, help='weight of symmetric loss')
parser.add_argument('--inner_steps', type=int, default=2, help='steps for inner optimization')
parser.add_argument('--outer_steps', type=int, default=1, help='steps for outer optimization')
parser.add_argument('--lr_adj', type=float, default=0.01, help='lr for training adj')
parser.add_argument('--symmetric', action='store_true', default=False,
            help='whether use symmetric matrix')
parser.add_argument('--adj_path', type=str, default="")
parser.add_argument('--filename', type=str, default="")
parser.add_argument('--norm', type=int, default=0, help='type of normalization')
parser.add_argument('--agg', type=str, default="mean", help='type of aggregation')


args = parser.parse_args()
args.secondary_seed = args.seed
print(args.agg)
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

if args.cuda:
    torch.cuda.manual_seed(args.secondary_seed)
if args.ptb_rate == 0:
    args.attack = "no"

print(args)

#np.random.seed(15) # Here the random seed is to split the train/val/test data, we need to set the random seed to be the same as that when you generate the perturbed graph

data = Dataset(root='/tmp/', name=args.dataset, setting='nettack')
adj, features, labels = data.adj, data.features, data.labels
#idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
splits = np.load('splits/{}_{}.npz'.format(args.dataset, args.seed))
idx_train, idx_val, idx_test = splits['train'], splits['val'], splits['test']

#print(idx_train)
if args.attack == 'no':
    perturbed_adj = adj

if args.attack == 'random':
    from deeprobust.graph.global_attack import Random
    attacker = Random()
    n_perturbations = int(args.ptb_rate * (adj.sum()//2))
    perturbed_adj = attacker.attack(adj, n_perturbations, type='add')

if args.attack == 'meta' or args.attack == 'nettack':
    perturbed_adj = sp.load_npz(args.adj_path)
np.random.seed(args.secondary_seed)
torch.manual_seed(args.secondary_seed)

prognn = GNN(
            #nnodes=features.shape[0],
            nfeat=features.shape[1],
            nhid=args.hidden,
            weight_decay=args.weight_decay,
            nclass=labels.max().item() + 1,
            dropout=args.dropout, device=device, norm=args.norm, agg=args.agg)

labels_ = np.copy(labels)
features_ = scipy.sparse.csr_matrix.copy(features) 
perturbed_adj, features, labels = preprocess(perturbed_adj, features, labels, preprocess_adj=False, device=device)
labes = labels.cuda()
perturbed_adj = perturbed_adj.cuda()

features = features.cuda()
prognn = prognn.cuda()

features[features > 0] = 1.

prognn.fit(features, perturbed_adj, labels, idx_train, idx_val, train_iters=args.epochs)
prognn.test(idx_test, args.filename)

