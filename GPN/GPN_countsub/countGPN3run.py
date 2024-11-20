import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (global_add_pool,GCNConv)
import numpy as np
from libs.utils import GraphCountDataset,GPNdesign
import scipy.io as sio
from libs.gnn import GPN, get_n_params
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd


"""
hyperparamètres
"""




operator = 'adj'



ep = 2000
nbrun = 3
batch_size = 64
step = .95
patience = 25
lr_init = .01
weight_decay = 0.







operator = 'adj'

output_dim = 1
num_layer = 1
nodes_dim = [2]*num_layer 
edges_dim = 2
decision_depth = 3
final_neuron = [64,32]
readout_type  = "sum"
level = "node"
relu = False
dropout = 0.0
normalize = False

# lamda1 = .1
# lamda2 = .1
# lamda3 = .1
# lamda4 = .1
# lamda5 = .1
# lamda6 = .1

"""
"""

typ = ["triangle","4-cycle","5_cycle","6_cycle","7_cycle"]
# select task, 0: triangle, 1: tailed_triangle 2: 4-cycle 3: trisquare  4: 5-cycle
ntask=0
transform = GPNdesign(operator = operator)


dataset = GraphCountDataset(root="dataset/subgraphcount/",pre_transform=transform)

print(dataset.data.edge_attr.shape)



a=sio.loadmat('dataset/subgraphcount/raw/randomgraph.mat')
trid=a['train_idx'][0]
ntrid = len(trid)
vlid = a['val_idx'][0]
nvlid=len(vlid)
tsid = a['test_idx'][0]
ntsid=len(tsid)
# normalize outputs
# dataset.data.y=(dataset.data.y/dataset[[i for i in trid]].data.y.std(0))
print(dataset[[i for i in trid]].data.y.std(0))





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

node_input_dim = dataset.num_features
edge_input_dim = dataset.num_edge_features

bVal_mae = []
bTest_mae = []

bvalsession = 1000
Loss = torch.nn.L1Loss()
for run in range(1,nbrun+1):
    train_loader = DataLoader(dataset[[i for i in trid]], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset[[i for i in vlid]], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset[[i for i in tsid]], batch_size=batch_size, shuffle=False)
    
    # select your model
    model = GPN(node_input_dim, edge_input_dim, output_dim,
                device, num_layer=num_layer, nodes_dim=nodes_dim,
                nedgeoutput=edges_dim, decision_depth=decision_depth,
                final_neuron=final_neuron, readout_type=readout_type,
                level=level, normalize=normalize).to(device)   
    
    lr = lr_init
    
    print('number of parameters:',get_n_params(model))
    
    # be sure PPGN's bias are initialized by zero
    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)         
                    
    # model.apply(weights_init)
    # torch.nn.init.ones_(model.conv1.fc1_2.weight[0,1])
    # torch.nn.init.ones_(model.conv1.fc1_3.weight[0,2])
    # torch.nn.init.ones_(model.conv1.fc1_6.weight[0,3])
    # torch.nn.init.ones_(model.fc1.weight)
    # torch.nn.init.ones_(model.fc2.weight)
    # torch.nn.init.ones_(model.conv1.conv1.weight)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=weight_decay)
    
        
    
    def train():
        model.train()
        
        L=0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            pre=model(data)
            
            lss= Loss(pre, data.y[:,ntask:ntask+1])
            
            
            
            lss.backward()
            optimizer.step()  
            L+=lss.item()
    
        return L/len(train_loader)
    
    def test():
        model.eval()
        yhat=[]
        ygrd=[]
        L=0
        for data in test_loader:
            data = data.to(device)
    
            pre=model(data)
            yhat.append(pre.cpu().detach())
            
            ygrd.append(data.y[:,ntask:ntask+1].cpu().detach())
            lss= Loss(pre, data.y[:,ntask:ntask+1])       
            L+=lss.item()
        yhat=torch.cat(yhat)
        ygrd=torch.cat(ygrd)
        testmae=mean_absolute_error(ygrd.numpy(),yhat.numpy())
        testmaerounded=mean_absolute_error(ygrd.numpy(),np.round(yhat.numpy()))
    
        Lv=0
        for data in val_loader:
            data = data.to(device)
            pre=model(data)
            lss= Loss(pre, data.y[:,ntask:ntask+1]) 
            Lv+=lss.item()    
        return L/len(test_loader), Lv/len(val_loader),testmae,testmaerounded
    
    
    
    bval=1000
    
    btest=0
    btestmae=0
    btestmaerounded=0
    Train_loss = []
    Val_loss = []
    Test_loss = []
    
    count = 0
    
    for epoch in tqdm(range(1, ep+1)):
        if count > patience:
            count = 0
            for g in optimizer.param_groups:
                lr = lr*step
                g['lr']= lr
        if lr < 1e-6:
            break
                
        trloss=train()
        test_loss,val_loss,testmae,testmaerounded = test()
        Train_loss.append(trloss)
        Val_loss.append(val_loss)
        Test_loss.append(test_loss)
        if bval>val_loss:
            torch.save(model.state_dict(), "save/countweight.dat")
            if bvalsession > val_loss:
                torch.save(model.state_dict(), "save/bestcountweight.dat")
                bvalsession = val_loss
            bval=val_loss
            btest=test_loss
            btestmae=testmae
            btestmaerounded=testmaerounded
            count = 0
        else:
            count +=1
        # print((run,epoch,lr,trloss,val_loss,test_loss,btest,btestmae))
        print('run : {:02d}, Epoch: {:02d}\nlr: {:.6f}, trloss: {:.6f},  Valloss: {:.6f},Testloss: {:.6f}, best test loss: {:.6f}, bestmae:{:.6f},bestmaerounded:{:.6f}'.format(run,epoch,lr,trloss,val_loss,test_loss,btest,btestmae,btestmaerounded))


    bTest_mae.append(btestmae)

    results = {           "btestmae" : bTest_mae}
    
    
    results_df = pd.DataFrame(results)
    
    results_df.to_csv("data/count"+typ[ntask]+"_GMN10run.dat", header = True, index = False)
    

    bt = np.array(bTest_mae)
    
 
    meanTest_mae = bt.mean()

    varTest = bt.var()
    
    print('mean test mae : {:.6f}, var test mae : {:.6f}'.format(meanTest_mae,varTest))

