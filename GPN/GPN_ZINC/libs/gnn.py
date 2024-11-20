import torch
import torch.nn as nn
from libs.layer_gnn import GPNLayer
from torch_geometric.nn import (global_add_pool,global_mean_pool, global_max_pool)
import libs.readout_gnn as ro

from time import time

def get_n_params(model):
    pp=0
    for p in model.parameters():
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

class GPN(nn.Module):
    def __init__(self,  node_input_dim, edge_input_dim, output_dim, device,
                 num_layer = 5, nodes_dim = [16,16,16,16,16],
                 nedgeoutput = 16, decision_depth = 3,final_neuron = [512,256],
                 readout_type  = "sum" ,level = "graph", relu = True, dropout = 0, normalize=False):
        
        super(GPN, self).__init__()
        
        self.num_layer = num_layer
        self.node_input_dim = node_input_dim
        self.edge_input_dim = edge_input_dim
        self.output_dim = output_dim
        self.nodes_dim = nodes_dim
        self.nedgeoutput = nedgeoutput
        self.decision_depth = decision_depth
        self.final_neuron = final_neuron
        self.readout_type = readout_type
        self.level =level
        self.conv = nn.ModuleList()
        self.device = device
        self.relu = relu
        
        if self.num_layer < 1:
            raise ValueError("Number of GNN layer must be greater than 1")
            
        if self.num_layer != len(self.nodes_dim):
            raise ValueError("Number of GNN layer must match length of nodes_dim."+
                             "\n num_layer = {}, neuron_dim length = {}"
                             .format(self.num_layer,len(self.nodes_dim)))
            
        if self.decision_depth != len(self.final_neuron) + 1:
            raise ValueError("Number of decision layer must match in decision depth" + 
                             "={}, final neuron dim + 1 = {}".format(self.decision_depth,len(self.final_neuron) + 1))
        
        
        if self.level == "node":
            self.readout = ro.node_level_readout

        elif self.level == "graph":
            if self.readout_type == "sum":
                self.readout = ro.graph_level_readout_sum
            elif self.readout_type == "mean":
                self.readout = ro.graph_level_readout_mean
            elif self.readout_type == "max":
                self.readout = ro.graph_level_readout_max
            else:
                raise ValueError("Invalid readout type")
        else:
            raise ValueError("Invalid level type, should be graph or node")
        
        self.mlp_input = torch.nn.Sequential(torch.nn.Linear(node_input_dim,256))
        for i in range(self.num_layer):
            if i == 0:
                
                self.conv.append(G2N2Layer(nedgeinput= edge_input_dim, nedgeoutput = self.nedgeoutput,
                                            nnodeinput= node_input_dim, nnodeoutput= self.nodes_dim[0], 
                                            device = self.device, normalize=normalize))

            else:
                self.conv.append(G2N2Layer( nedgeinput = edge_input_dim, nedgeoutput = self.nedgeoutput,
                                            nnodeinput= self.nodes_dim[i-1], nnodeoutput= self.nodes_dim[i], 
                                            device = self.device, normalize=normalize))
        num_feat = 0
        for n in self.nodes_dim:
            num_feat +=n
        self.fc = nn.ModuleList( [torch.nn.Linear(self.nodes_dim[-1],self.final_neuron[0])])
        for i in range(self.decision_depth-2):
            self.fc.append(torch.nn.Linear(self.final_neuron[i], self.final_neuron[i+1]))


        self.fc.append(torch.nn.Linear(self.final_neuron[-1], self.output_dim))
        
        self.dropout = torch.nn.Dropout(p = dropout)
    
    def forward(self,data):
        x = data.x
        # x = self.mlp_input(x)
        out = 0
        edge_index=data.edge_index2
        C=data.edge_attr
        batch_node = x[:,0]*0.
        
        if self.relu:
            for i,l in enumerate(self.conv):
                x=(l(x, edge_index, C,batch_node))
                x = self.dropout(x)
                if i < self.num_layer - 1:
                    x = torch.relu(x)
                out += x
        else:
            for l in self.conv:
                x=(l(x, edge_index, C,batch_node))
                x = self.dropout(x)
                # print(x)
                out += x
        x = self.readout(out,data.batch)
        for i in range(self.decision_depth-1):
            x = torch.relu(self.fc[i](x))
        return self.fc[-1](x) 
        
