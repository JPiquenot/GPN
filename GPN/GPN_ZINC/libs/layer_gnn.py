from torch_geometric.typing import OptTensor
import math
import torch
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from time import time
import numpy as np


def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


class Conv_agg(torch.nn.Module):
    """
    """
    def __init__(self, in_channels, out_channels, device, K=1,bias=True):
        super(Conv_agg, self).__init__()

        assert K > 0       
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.shapetensor = torch.zeros((K,1)).to(device)
          

        self.weight = Parameter(torch.Tensor(K, in_channels, out_channels))
       
        if bias:
           self.bias = Parameter(torch.Tensor(out_channels))
        else:
           self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)


    def forward(self, h, X,edge_index,batch_node):
        """"""
        
        zer = torch.unsqueeze(batch_node*0.,0)

        resx = torch.matmul(torch.unsqueeze(torch.matmul(self.shapetensor,zer),2),zer)
        resx[:,edge_index[0],edge_index[1]] = X.T
        
        res = torch.matmul(resx,h)
        res = torch.matmul(res,self.weight).max(0)[0]         

        if self.bias is not None:
            res += self.bias

        return res
    
    
    def __repr__(self):
        return '{}({}, {}, K={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,self.weight.size(0))

class GPNLayer(torch.nn.Module):

    def __init__(self, nedgeinput,nedgeoutput,nnodeinput,nnodeoutput,device, normalize = True, dropout=0):
        super(GPNLayer, self).__init__()

        self.nedgeinput  = nedgeinput
        self.nnodeinput  = nnodeinput
        self.nnodeoutput = nnodeoutput
        self.shapetensor = torch.zeros(nedgeinput,1).to(device)

        self.normalize = normalize

        self.L1 = torch.nn.Linear(nedgeinput, nedgeinput,bias=False)
        self.L2 = torch.nn.Linear(nedgeinput, nedgeinput,bias=False)

        self.mlpedge_Normalized = torch.nn.Sequential(torch.nn.Linear(2*nedgeinput,4*nedgeinput,bias=False),
                                           torch.nn.ReLU(),
                                           
                                           #torch.nn.BatchNorm1d(4*nedgeinput, track_running_stats=False),  # batch normalisation après la couche ReLU
                                           torch.nn.LayerNorm(4*nedgeinput),
                                           
                                           torch.nn.Linear(4*nedgeinput,nedgeoutput,bias=False)
                                          )

        self.mlpedge = torch.nn.Sequential(torch.nn.Linear(2*nedgeinput,4*nedgeinput,bias=False),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(4*nedgeinput,nedgeoutput,bias=False)
                                          )
        
        self.mlpnode_Normalized = torch.nn.Sequential(torch.nn.Linear(nnodeoutput ,4*nnodeinput,bias=False),
                                           torch.nn.ReLU(),
                                           
                                           #torch.nn.BatchNorm1d(4*nnodeinput, track_running_stats=False),  # batch normalisation après la couche ReLU
                                           torch.nn.LayerNorm(4*nnodeinput),
                                           
                                           torch.nn.Linear(4*nnodeinput ,nnodeoutput,bias=False)
                                          )

        self.mlpnode = torch.nn.Sequential(torch.nn.Linear(nnodeoutput ,4*nnodeinput,bias=False),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(4*nnodeinput ,nnodeoutput,bias=False)
                                          )
        self.dropout = torch.nn.Dropout(dropout)
        self.agg = Conv_agg(nnodeinput, nnodeoutput, device, K=nedgeoutput,bias = False)

    def forward(self, x, edge_index, C, batch_node):      

        if self.normalize:
            x = torch.nn.functional.normalize(x, dim=1)
            C = torch.nn.functional.normalize(C, dim=1)
            C = torch.nn.functional.normalize(C, dim=0)

        tmp=torch.cat([ (C) , (self.L1(C))* (self.L2(C)) ],1)

        if self.normalize : 
            Cout = self.mlpedge_Normalized(tmp)
            xout=self.mlpnode_Normalized((self.agg(x, Cout, edge_index, batch_node)))
            #xout=(self.agg(x, Cout, edge_index, batch_node))
            #xout=torch.nn.functional.normalize(xout, dim=1)
            xout = self.dropout(xout)
            
        else:
            Cout = self.mlpedge(tmp)
            xout=self.mlpnode((self.agg(x, Cout, edge_index, batch_node)))
            #x = self.dropout(x)
            #xout=(self.agg(x, Cout, edge_index, batch_node))
            
        return xout

