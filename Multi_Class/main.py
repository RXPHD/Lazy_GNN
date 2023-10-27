import time

import argparse
import os.path as osp
import numpy as np
import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.nn import Linear
from torch.utils.data import DataLoader

import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected
from torch_geometric.loader import DataLoader
from torch_geometric.loader import NeighborLoader
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric_autoscale import get_data
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from lazy_model import Net
from preprocessing import metis, permute, SubgraphLoader
from args import get_args
from torch_geometric.nn.conv.gcn_conv import gcn_norm


def train(model, optimizer, train_loader, device, args):
    model.train()
    for batch in train_loader:
        x = batch.x.to(device)
        adj_t = batch.adj_t.to(device)
        id = batch.n_id.to(device)
        optimizer.zero_grad()
        train_target = batch.train_mask[:batch.batch_size].to(device)  ###get target nodes
        y = batch.y[:batch.batch_size][train_target].to(device)
        out = model(x, adj_t, id, batch.batch_size, args.K_train, args.alpha, args.beta_train, args.theta, device)[:batch.batch_size][train_target]
        loss = F.nll_loss(out, y)
        loss.backward()
        optimizer.step()
    return float(loss)


@torch.no_grad()
def test(model, data, device, args):
    model.eval()
    mlp = model.inference(data.x, device).cpu()#.to(device)
    
    z_pred = mlp
    for i in range(args.K_val_test):
        #APPNP for inference
        data.adj_t = data.adj_t#.to(device)
        z_pred = (1 - args.alpha) * (data.adj_t @ z_pred) + args.alpha * mlp


    out = F.softmax(z_pred, dim=1)
    y_hat = out.argmax(dim=-1)
    y = data.y.to(y_hat.device)
    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((y_hat[mask] == y[mask]).sum()) / int(mask.sum()))
    return accs[0], accs[1], accs[2]


# @hydra.main(config_path='conf', config_name='config')
def main(args):
    
    root='/tmp/datasets'
    data, in_channels, out_channels = get_data(root, args.dataset)
    print(data)
    perm, ptr = metis(data.adj_t, num_parts=args.num_parts, log=True)  #### clustering
    data = permute(data, perm, log=True)
    data.n_id = torch.arange(data.num_nodes) ### assign index to nodes
    
    data.adj_t = data.adj_t.set_diag()
    data.adj_t = gcn_norm(data.adj_t, add_self_loops=False)
    
    train_loader = SubgraphLoader(data, ptr, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0,
                                  persistent_workers=False)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = Net(in_channels, args.hidden, out_channels, args.num_layers, data.num_nodes, args.dropout)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    best = 0
    best_val = 0
    model.reset_parameters()
    for epoch in range(args.epochs):
        loss = train(model, optimizer, train_loader, device, args)
        train_acc, val_acc, test_acc = test(model, data, device, args)
        if val_acc > best_val:
            best_val = val_acc
            best = test_acc
        print(f'Epoch: {epoch:02d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
            f'Test: {test_acc:.4f}, Best: {best:.4f}')
            
    
    
if __name__ == "__main__":
    args = get_args()
    main(args)