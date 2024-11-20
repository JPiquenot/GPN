import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import (global_add_pool,global_mean_pool, global_max_pool)





def node_level_readout(x,batch_node):
    return x


def graph_level_readout_sum(x,batch_node):
    
    return global_add_pool(x,batch_node)

def graph_level_readout_max(x,batch_node):
    
    return global_max_pool(x,batch_node)

def graph_level_readout_mean(x,batch_node):
    
    return global_mean_pool(x,batch_node)