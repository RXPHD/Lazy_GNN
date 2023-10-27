import argparse
import os.path as osp
import numpy as np
import time
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch.nn import Linear
import scipy.sparse as sp

import torch_geometric
import torch_geometric.transforms as T


##### define lazy propagation

class Lazy_Prop(torch.autograd.Function):

    @staticmethod
    def forward(self, x: Tensor, adj_matrix, id, size, K: int, alpha: float, beta: float, theta: float, 
                equ_preds: torch.FloatTensor, equ_grad: torch.FloatTensor, device, **kwargs):

        self.size = size
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.adj_matrix = adj_matrix
        self.device = device
        self.adj_matrix = self.adj_matrix.to(self.device)
        z_sam = torch.zeros_like(x)
        g_sam = torch.zeros_like(x)
        
        # use memory
        z_sam = equ_preds[id]
        g_sam = equ_grad[id]
        
        z_sam = z_sam.to(self.device)
        g_sam = g_sam.to(self.device)
       
        self.save_for_backward(z_sam, g_sam)
        
         # forward pass

        z = torch.zeros_like(x)
        
        ### forward lazy propagation & momentum connection

        if torch.equal(z_sam[:self.size], torch.zeros_like(z_sam)[:self.size].to(self.device)):
            z = x
        else:
            z = (1-self.beta)*z_sam + self.beta*x #target nodes

        ### aggragation
        
        for i in range(self.K):
            z = (1 - self.alpha) * (self.adj_matrix @ z) + self.alpha * x
        return z

    @staticmethod
    def backward(self, grad_output):
        z_sam, g_sam = self.saved_tensors
        
        g = torch.zeros_like(g_sam)

        ###backward lazy propagation & momentum connection
        
        if torch.equal(g_sam[:self.size], torch.zeros_like(g_sam)[:self.size].to(self.device)):
            g = grad_output
        else:
            g = (1-self.theta)*g_sam + self.theta*grad_output
   
        for j in range(self.K):
            g = (1 - self.alpha) * (self.adj_matrix @ g) + self.alpha * grad_output

        g[self.size:] = 0  ###use gradients of target nodes 
        return g, None, None, None, None, None, None, None, None, None, None
    
    
    
    
class Net(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, num_layers, num_nodes, dropout, **kwargs):
        super(Net, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(num_features, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, num_classes))
        self.prop1 = Lazy_Prop()
        self.g_mem = torch.zeros(num_nodes, num_classes)
        self.z_mem_tr = torch.zeros(num_nodes, num_classes)
        self.num_classes = num_classes
        self.num_nodes = num_nodes
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()


    def forward(self, x, adj, id, size, K, alpha, beta, theta, device):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        def backward_hook(grad):   ### lazy backprop
            self.g_mem = self.g_mem.to(device)
            self.g_mem[id[:size]] = grad[:size].clone().detach()
            return grad
        x.register_hook(backward_hook)
        z_out = self.prop1.apply(x, adj, id, size, K, alpha, beta, theta, self.z_mem_tr, self.g_mem, device) ### lazy forward
        self.z_mem_tr = self.z_mem_tr.to(device)
        self.z_mem_tr[id[:size]] = z_out[:size].clone().detach()  ###cache into memory
        out = F.log_softmax(z_out, dim=1)
        return out
    

    @torch.no_grad()
    def inference(self, x, device):
        self.eval()
        x = x.to(device)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x