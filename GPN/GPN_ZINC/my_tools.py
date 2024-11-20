# Importation des bibliothèques requises
import torch
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as transforms
from torch_geometric.transforms import NormalizeFeatures

import numpy as np
from tqdm import tqdm
from libs.gnn import GPN
import ast
from libs.utils import GPNdesign
import random
import os
import pandas as pd
from copy import deepcopy

from torch.optim.lr_scheduler import ReduceLROnPlateau

import warnings
warnings.filterwarnings('ignore')



# Fonctions pour lancer l'entraînement et les tests
# ********************************************************************************

# Fonction de régularisation L1
def l1_regularization(model, l1_lambda):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return l1_lambda * l1_norm


def train(model, train_loader, optimizer, device, l1_lambda):
    model.train()
    ntrid = len(train_loader.dataset)
    L = 0

    for data in train_loader:
        data = data.to(device)
        pre = model(data)
        
        lss = torch.abs(pre - data.y).sum()
        if l1_lambda > 0:
            lss += l1_regularization(model, l1_lambda)
            
        optimizer.zero_grad()
        lss.backward()
        optimizer.step()
        L += lss.item()

    return L / ntrid

def test(model, test_loader, val_loader, device):
    model.eval()
    
    nvlid = len(val_loader.dataset)
    ntsid = len(test_loader.dataset)

    yhat, ygrd = [], []
    L, Lv = 0, 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            pre = model(data)
            yhat.append(pre.cpu())
            ygrd.append(data.y.cpu())
            L += torch.abs(pre - data.y).sum().item()
        for data in val_loader:
            data = data.to(device)
            pre = model(data)
            Lv += torch.abs(pre - data.y).sum().item()
    yhat = torch.cat(yhat, 0)
    ygrd = torch.cat(ygrd, 0)
    testmae = np.abs(ygrd.numpy() - yhat.numpy()).mean()
    return L / ntsid, Lv / nvlid, testmae


def train_and_test(model, hyperparams, train_loader, val_loader, test_loader):
    
    patience = hyperparams['patience']
    lr = hyperparams['lr']
    ep = hyperparams['ep']
    step = hyperparams['step']
    epsi = hyperparams['epsi']
    device = hyperparams['device']
    l2_reg = hyperparams['l2_reg']
    l1_reg = hyperparams['l1_reg']

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=step, patience=patience, min_lr=epsi)

    bval, btest, btestmae = float('inf'), 0, 0
    Train_loss, Val_loss, Test_loss, bTest_mae = [], [], [], []

    for epoch in tqdm(range(1, ep + 1)):
        current_epoch = epoch
                
        trloss = train(model, train_loader, optimizer, device, l1_reg)

        test_loss, val_loss, testmae = test(model, test_loader, val_loader, device)
        Train_loss.append(trloss)
        Val_loss.append(val_loss)
        Test_loss.append(test_loss)

        scheduler.step(val_loss)

        if bval > val_loss:
            torch.save(model.state_dict(), "save/zinc.dat")
            bval, btest, btestmae = val_loss, test_loss, testmae
            bTest_mae.append(btestmae)
        else:
            bTest_mae.append(bTest_mae[-1])
        
        res_message = f'Epoch: {epoch:02d} ===> '
        res_message += f'trloss: {trloss:.6f}, val_loss: {val_loss:.6f}, test_loss: {test_loss:.6f}, '
        res_message += f'bestmae: {btestmae:.6f}'
        print(res_message)
        
    perf_results = {
        'Train_loss': Train_loss,
        'Val_loss': Val_loss,
        'Test_loss': Test_loss,
        'bTest_mae': bTest_mae
    }

    return perf_results


# Fonction d'initialisation des hyperpatamètre à des valeurs par défaut
# Cette fonction renvoi un dictionnaire que l'on peut modifier pour des personnalisations
# ***************************************************************************************
def init_hParams():
    hyperparams = {
            # Paramètrage du prétraitement des données
            'normalize': True,  
            'hot_encoding': True,  
            
            # paramètrage de l'optilisation
            'lr': 0.0001,  
            'patience': 25,  
            'step': 0.90,  
            'epsi': 0.000001,

            'l1_reg': 0.01,  # Coefficient de régularisation L1
            'l2_reg': 0.01,  # Coefficient de régularisation L2
            
            # Paramètrage de la gestion du processus d'entrainement
            'batch_size': 64,
            'ep': 2,  
            
            # Paramérage du modèle de GNN
            'ntask': 0,   # Indice de la tâche, à ajuster selon le besoin
            'depth': 5, 
            'operator': 'adj',  # Opérateur de transformation des données
            'output_dim': 1,  
            'num_layer': 5,  
            'nodes_dim': [8,8,8,8,8],  
            'edges_dim': 11,  
            'decision_depth': 2,
            'final_neuron': [128],  
            'readout_type': 'max', 
            'level': 'graph',
            
            # Paramétrage des condition d'exécution
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'seed_number' : np.random.randint(1000) 
        }  
    return hyperparams  


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Fonction pour charger et transformer les données et créer une instance du modèle G2N2
# ********************************************************************************

def load_and_transform_data(h, datasetPath="dataset/ZINC/"):
    
    transform = GPNdesign(operator=h['operator'], depth=h['depth'])

    train_dt = ZINC(root=datasetPath, pre_transform=transform, split='train', subset=True)
    valid_dt = ZINC(root=datasetPath, pre_transform=transform, split='val',   subset=True)
    test_dt = ZINC(root=datasetPath,  pre_transform=transform, split='test',  subset=True)
   
    transform = GPNdesign(operator=h['operator'], depth=h['depth'])

    train_dt = ZINC(root=datasetPath, pre_transform=transform, split='train', subset=True)
    valid_dt = ZINC(root=datasetPath, pre_transform=transform, split='val',   subset=True)
    test_dt = ZINC(root=datasetPath,  pre_transform=transform, split='test',  subset=True)


    train_dt.data.y = train_dt.data.y.unsqueeze(1)
    valid_dt.data.y = valid_dt.data.y.unsqueeze(1)
    test_dt.data.y = test_dt.data.y.unsqueeze(1)

    train_loader = DataLoader(train_dt, batch_size=h['batch_size'], shuffle=True)
    val_loader = DataLoader(valid_dt, batch_size=h['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dt, batch_size=h['batch_size'], shuffle=False)

    h['node_input_dim'] = train_dt.num_features
    h['edge_input_dim'] = train_dt.num_edge_features

    return train_loader, val_loader, test_loader


def load_transform_hot_encod_data(h, datasetPath="datasetHotEncod/ZINC_New/"):

    class OneHotNodeFeatures(object):
        def __init__(self, node_types):
            self.c = node_types

        def __call__(self, data):
            n = data.x.shape[0]
            node_encoded = torch.zeros((n, self.c), dtype=torch.float32)
            node_encoded.scatter_(1, data.x.long(), 1)
            data.x = node_encoded
            return data
    
    # Définir le nombre de types de noeuds et d'arêtes
    node_types = 21

    # Créer une instance de la classe OneHotNodeEdgeFeatures
    one_hot_transform = OneHotNodeFeatures(node_types)

    transform = transforms.Compose([
        G2N2design(operator=h['operator'], QM9=False, depth=h['depth']),
        one_hot_transform
    ])

    train_dt = ZINC(root=datasetPath, pre_transform=transform, split='train', subset=True)
    valid_dt = ZINC(root=datasetPath, pre_transform=transform, split='val', subset=True)
    test_dt  = ZINC(root=datasetPath, pre_transform=transform, split='test', subset=True)

    train_dt.data.y = train_dt.data.y.unsqueeze(1)
    valid_dt.data.y = valid_dt.data.y.unsqueeze(1)
    test_dt.data.y = test_dt.data.y.unsqueeze(1)

    train_loader = DataLoader(train_dt, batch_size=h['batch_size'], shuffle=True)
    val_loader = DataLoader(valid_dt, batch_size=h['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dt, batch_size=h['batch_size'], shuffle=False)

    h['node_input_dim'] = train_dt.num_node_features
    h['edge_input_dim'] = train_dt.num_edge_features

    return train_loader, val_loader, test_loader
    

def generate_Model_Name(file_path):
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        last_model_number = existing_df['model_name'].str.extract(r'M_(\d+)').astype(int).max().item()
        new_model_number = last_model_number + 1
    else:
        existing_df = pd.DataFrame()
        new_model_number = 1

    new_model_name = f"M_{new_model_number}"
    return existing_df, new_model_name    


# Fonction pour sauvegarder les hyperparamètres et les résultats dans un fichier CSV
# **********************************************************************************

def save_perf(hyperparams, perf_results, file_path = "res/results.csv"):
    
    existing_df, new_model_name = generate_Model_Name(file_path)
    
    # Créer un DataFrame à partir des résultats de performance
    df = pd.DataFrame.from_dict(perf_results)
    
    # Ajouter les hyperparamètres et le nom du modèle
    for key, value in hyperparams.items():
        df[key] = [value] * len(df)
    
    df['model_name'] = new_model_name

    if not existing_df.empty:
        combined_df = pd.concat([existing_df, df], ignore_index=True)
    else:
        combined_df = df    
        
    # Sauvegarder le DataFrame combiné
    combined_df.to_csv(file_path, index=False)
    print(f"\nLes hyperparamètres et les résultats ont été sauvegardés dans {file_path}")
    
    return df


# Fonction qui vérifie si les hyperparamètres fournis ont déjà 
# été testés c'est à dire existent dans le fichier CSV des résultats
# ******************************************************************

def tested_before(h, res_file="res/results.csv"):
    
    # Si le fichier des résultats n'existe pas cela veut dire qu'aucune
    # expérience n'a été menée auparavant, donc on retounre False
    if not os.path.exists(res_file):
        return False

    # Charger les résultats des tests précédents depuis le fichier CSV
    hyperparams = deepcopy(h)
    df = pd.read_csv(res_file)[hyperparams.keys()]
        
    hyperparams["nodes_dim"] = str(h['nodes_dim'])
    hyperparams["final_neuron"] = str(hyperparams['final_neuron'])
    
    if torch.cuda.is_available():
        hyperparams["device"] = 'cuda'
    else :
        hyperparams["device"] = 'cpu'
        
    L1 = list(hyperparams.values())

    for i in range(df.shape[0]):
        L2 = list(df.iloc[i, :].values)
        if L1==L2:
            return True
    return False


