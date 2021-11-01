import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import DataLoader
from torch_geometric.nn import (NNConv, graclus, max_pool, max_pool_x,ARMAConv,global_mean_pool,GATConv,ChebConv,GCNConv)
from Bern import BernConv
from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm

class GPR_prop(MessagePassing):
    def __init__(self, K, alpha=0.1, Init='Random', Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0*np.ones(K+1)
            TEMP[-1] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)
class GPRNet(torch.nn.Module):
    def __init__(self,K=10):
        super(GPRNet, self).__init__()
        self.lin1 = Linear(1, 32)
        self.lin2 = Linear(32, 64)
        self.prop1 = GPR_prop(K)
        self.fc2 = torch.nn.Linear(64, 1)

    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, data):
        x=data.x_tmp
        edge_index=data.edge_index

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))

        x = self.prop1(x, edge_index)
        return self.fc2(x)


class ARMANet(nn.Module):
    def __init__(self):
        super(ARMANet, self).__init__()
        self.conv1 = ARMAConv(1,32,1,1,False,dropout=0)
        self.conv2 = ARMAConv(32,64,1,1,False,dropout=0)
        self.fc2 = torch.nn.Linear(64, 1)

    def forward(self, data):

        x=data.x_tmp
        edge_index=data.edge_index
        
        x = F.relu(self.conv1(x, edge_index))  
        x = F.relu(self.conv2(x, edge_index))      
        return self.fc2(x)


class GcnNet(nn.Module):
    def __init__(self):
        super(GcnNet, self).__init__()

        self.conv1 = GCNConv(1, 32, cached=False)
        self.conv2 = GCNConv(32, 64, cached=False) 
        
        self.fc2 = torch.nn.Linear(64, 1) 

    def forward(self, data):

        x=data.x_tmp
        edge_index=data.edge_index
        
        x = F.relu(self.conv1(x, edge_index))  
        x = F.relu(self.conv2(x, edge_index))      
        return self.fc2(x)

class GatNet(nn.Module):
    def __init__(self):
        super(GatNet, self).__init__()
        self.conv1 = GATConv(1, 8, heads=4,concat=True, dropout=0.0)  
        self.conv2 = GATConv(32, 8, heads=8,concat=True, dropout=0.0) 
        
        self.fc2 = torch.nn.Linear(64, 1) 

    def forward(self, data):
        x=data.x_tmp
        x = F.elu(self.conv1(x, data.edge_index))
        x = F.elu(self.conv2(x, data.edge_index)) 

        return self.fc2(x) 

class ChebNet(nn.Module):
    def __init__(self,K=3):
        super(ChebNet, self).__init__()
        
        self.conv1 = ChebConv(1, 32,K)    
        self.conv2 = ChebConv(32, 32,K) 
        self.fc2 = torch.nn.Linear(32, 1) 

    def forward(self, data):
        x=data.x_tmp
        edge_index=data.edge_index        
        x = F.relu(self.conv1(x, edge_index))   
        x = F.relu(self.conv2(x, edge_index))  
           
        return self.fc2(x) 


class BernNet(nn.Module):
    def __init__(self,K=10):
        super(BernNet, self).__init__()

        self.conv1 = BernConv(1, 32, K)
        self.conv2 = BernConv(32, 64, K)

        self.fc2 = torch.nn.Linear(64, 1)
        self.coe = Parameter(torch.Tensor(K+1))
        self.reset_parameters()

    def reset_parameters(self):
        self.coe.data.fill_(1)

    def forward(self, data):
        x=data.x_tmp
        edge_index=data.edge_index
        
        x = F.relu(self.conv1(x, edge_index,self.coe))  
        x = F.relu(self.conv2(x, edge_index,self.coe))       
        return self.fc2(x) 