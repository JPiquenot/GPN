import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from ogb.graphproppred import GraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (global_add_pool,global_mean_pool,GCNConv, global_max_pool)
import numpy as np
from libs.utils import HivDataset,G2N2design
from sklearn.metrics import roc_auc_score,confusion_matrix
import pandas as pd
import gzip
import matplotlib.pyplot as plt

from libs.gnn import G2N2, get_n_params
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from tqdm import tqdm
import pickle


"""
hyperparamètres
"""




 
lr_init = 0.0005
patience = 5
step = .95
epsi = 1e-6
ep = 75
batch_size=64
weight_decay = 0




operator = 'adj'

output_dim = 1
num_layer = 1
nodes_dim = [16]*num_layer 
edges_dim = 32
decision_depth = 2
final_neuron = [128]
readout_type  = "sum"
level = "graph"
dropout = 0.8
relu = False

"""
"""





"""
"""


mol = GraphPropPredDataset(name = "ogbg-molhiv", root = 'dataset/')


evaluator = Evaluator('ogbg-molhiv')

transform = G2N2design(operator = operator)

dataset = HivDataset(root="dataset/ogbg_molhiv/",pre_transform=transform)
# norm = torch.linalg.norm(dataset.data.edge_attr2)
# norm2 = torch.linalg.norm(dataset.data.x)
# dataset.data.edge_attr2 = dataset.data.edge_attr2/norm
# dataset.data.x = dataset.data.x/norm2


res = []
for i in range(dataset.data.x.shape[1]):
    
    res.append(F.one_hot(dataset.data.x[:,i].type(torch.int64)).type(torch.float32))
dataset.data.x = torch.cat(res,1)


# res = [dataset.data.edge_attr[:,0:2]]
# for i in range(2,dataset.data.edge_attr.shape[1]):
#     res.append(F.one_hot(dataset.data.edge_attr[:,i].type(torch.int64)).type(torch.float32))
# dataset.data.edge_attr = torch.cat(res,1)






split_idx = mol.get_idx_split() 


tr_list = split_idx['train']
vl_list = split_idx['valid']
ts_list = split_idx['test']


nb_true = dataset[tr_list].data.y.sum()
prop = (len(dataset[tr_list].data.y)-nb_true)/nb_true

train_loader = DataLoader(dataset[tr_list], batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(dataset[vl_list], batch_size=batch_size, shuffle=False)
test_loader = DataLoader(dataset[ts_list], batch_size=batch_size, shuffle=False)

trid=len(tr_list)
vlid=len(vl_list)
tsid=len(ts_list)


node_input_dim = dataset.num_features
edge_input_dim = dataset.num_edge_features



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
auc_test = []
auc_val = []

for i in range(10):

    model = G2N2(node_input_dim, edge_input_dim, output_dim, device,
                 num_layer = num_layer, nodes_dim = nodes_dim, decision_depth =decision_depth,
                 nedgeoutput = edges_dim,final_neuron = final_neuron,
                 readout_type  = readout_type ,level = level,relu = relu,dropout = dropout).to(device)
    param = get_n_params(model)
    print('number of parameters:', param)
    
    lr = lr_init
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay = weight_decay)
    
    
    
    # FL = nn.BCEWithLogitsLoss(pos_weight = prop,reduction = 'sum')
    FL = nn.BCEWithLogitsLoss(pos_weight = torch.tensor(1.),reduction ='sum')
    # FL = torch.nn.MSELoss()
    
    def train(epoch):
        model.train()
        L=0
        yhat=[]
        ygrd=[]
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            
            pre=model(data)
            yhat.append(pre.cpu().detach())
            ygrd.append(data.y.cpu().detach())
            lss= FL(pre, data.y) 
            
            lss.backward()
            optimizer.step()  
            L+=lss.item()
    
        yhat=torch.cat(yhat, dim = 0)
        ygrd=torch.cat(ygrd, dim = 0)
        
        input_dict = {"y_true": ygrd.numpy(), "y_pred": torch.sigmoid(yhat).numpy()}
        trroc = evaluator.eval(input_dict)['rocauc']
        print(confusion_matrix(ygrd.numpy(),(yhat>0).numpy()),trroc)
        return L/trid, trroc
    
    def test():
        model.eval()
        with torch.no_grad():
            yhat=[]
            ygrd=[]
            L=0
            for data in test_loader:
                data = data.to(device)
        
                pre=model(data)
                yhat.append(pre.cpu().detach())
                ygrd.append(data.y.cpu().detach())
                lss= FL(pre, data.y)       
                L+=lss.item()
            yhat=torch.cat(yhat, dim = 0)
            ygrd=torch.cat(ygrd, dim = 0)
            print(torch.cat([torch.sigmoid(yhat[190:200]),ygrd[190:200]],1))
            input_dict = {"y_true": ygrd.numpy(), "y_pred": torch.sigmoid(yhat).numpy()}
            testroc=evaluator.eval(input_dict)['rocauc']
            print(confusion_matrix(ygrd.numpy(),(yhat>0).numpy()),testroc)
            
            yhat=[]
            ygrd=[]
        
            Lv=0
            for data in valid_loader:
                data = data.to(device)
                pre=model(data)
                yhat.append(pre.cpu().detach())
                ygrd.append(data.y.cpu().detach())
                lss= FL(pre, data.y) 
                Lv+=lss.item()  
            yhat=torch.cat(yhat, dim = 0)
            ygrd=torch.cat(ygrd, dim = 0)
            
            input_dict = {"y_true": ygrd.numpy(), "y_pred": torch.sigmoid(yhat).numpy()}
            valroc=evaluator.eval(input_dict)['rocauc']
            print(confusion_matrix(ygrd.numpy(),(yhat>0).numpy()),valroc)
        return L/tsid, Lv/vlid,testroc, valroc
    
    bval=0
    btest=0
    btestroc=0
    bv_loss = 1000
    
    
    early_stop = 0
    
    Train_loss = []
    Val_loss = []
    Test_loss = []
    Train_roc = []
    Val_roc = []
    Test_roc = []
    for epoch in tqdm(range(1, ep+1)):
        trloss, trroc =train(epoch)
        test_loss,val_loss,testroc, valroc = test()
        Train_loss.append(trloss)
        Val_loss.append(val_loss)
        Test_loss.append(test_loss)
        Train_roc.append(trroc)
        Val_roc.append(valroc)
        Test_roc.append(testroc)
        early_stop += 1
        if bval<valroc:
            torch.save(model.state_dict(), "save/molhivlearnfilter.dat")
            bval=valroc
            bv_loss = val_loss
            btestroc=testroc
            early_stop = 0
        if early_stop > patience:
            early_stop = 0
            for g in optimizer.param_groups:
                lr = lr*step
                g['lr']= lr
        if lr < epsi:
            ep = epoch
            break
            
         
        print('Lr : {:.6f}, Epoch: {:02d},\n trloss: {:.6f},  Valloss: {:.6f}, Testloss: {:.6f}, best val roc: {:.6f}, bestroc:{:.6f}'.format(lr,epoch,trloss,val_loss,test_loss,bval,btestroc))
    auc_test.append(Test_roc)
    auc_val.append(Val_roc)
    
    results = {  "auctest"  : l for i, l in enumerate(auc_test)}
    
    res2 = {  "aucval" : l for i, l in enumerate(auc_val)}
    
    results = {**results,**res2}
    results_df = pd.DataFrame(results)
    
    results_df.to_csv("data/ogbtest"+str(i)+".dat", header = True, index = False)

# X = range(1, ep+1)
# fig, ax = plt.subplots(1,2)

# ax[0].plot(X,Val_loss, label = "Valid")
# ax[0].plot(X,Train_loss, label = "train")
# ax[0].plot(X,Test_loss, label = "test")
# ax[0].set_title("Loss")


# ax[1].plot(X,Val_roc, label = "Valid")
# ax[1].plot(X,Train_roc, label = "train")
# ax[1].plot(X,Test_roc, label = "test")
# ax[1].set_title("AUC")
# plt.legend()

