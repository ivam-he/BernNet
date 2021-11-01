from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor, PairTensor
import math
import torch
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, matmul, fill_diag, sum, mul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops,get_laplacian,remove_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import torch.nn.functional as F
from scipy.special import comb

class BernConv(MessagePassing):

    def __init__(self, in_channels, out_channels, K,bias=True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(BernConv, self).__init__(**kwargs)
        assert K > 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.K=K

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(-2) + self.weight.size(-1)))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.fill_(0)

    def forward(self,x,edge_index,coe,edge_weight=None):

        TEMP=F.relu(coe)

        #L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight,normalization='sym', dtype=x.dtype, num_nodes=x.size(self.node_dim))

        #2I-L
        edge_index2, norm2 = add_self_loops(edge_index1,-norm1,fill_value=2.,num_nodes=x.size(self.node_dim))

        tmp=[]
        tmp.append(x)
        for i in range(self.K):
                x=self.propagate(edge_index2,x=x,norm=norm2,size=None)
                tmp.append(x)

        out=(comb(self.K,0)/(2**self.K))*TEMP[0]*tmp[self.K]

        for i in range(self.K):
                x=tmp[self.K-i-1]
                x=self.propagate(edge_index1,x=x,norm=norm1,size=None)
                for j in range(i):
                        x=self.propagate(edge_index1,x=x,norm=norm1,size=None)

                out=out+(comb(self.K,i+1)/(2**self.K))*TEMP[i+1]*x

        out=out@self.weight
        if self.bias is not None:
                out+=self.bias
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={}, normalization={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.weight.size(0), self.normalization)

