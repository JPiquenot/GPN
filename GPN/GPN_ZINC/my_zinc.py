import torch.nn.functional as F
from libs.utils import GPNdesign
from libs.gnn import GPN, get_n_params
import pandas as pd
from my_tools import save_perf, seed_everything, tested_before, init_hParams, train_and_test, load_transform_hot_encod_data
import numpy as np
import torch

from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as transforms
from torch_geometric.transforms import NormalizeFeatures

from tqdm import tqdm
import ast
import random
import os
from copy import deepcopy
from torch.optim.lr_scheduler import ReduceLROnPlateau

import warnings
warnings.filterwarnings('ignore')




# Fonction qui crée le modèle à partir d'une liste de paramètres
# **************************************************************
def generate_model(h):
    node_input_dim = h['node_input_dim']
    edge_input_dim = h['edge_input_dim']
    output_dim = h['output_dim']
    device = h['device']
    num_layer = h['num_layer']
    nodes_dim = h['nodes_dim']
    edges_dim = h['edges_dim']
    decision_depth = h['decision_depth']
    final_neuron = h['final_neuron']
    readout_type = h['readout_type']
    level = h['level']
    normalize = h['normalize']

    return GPN(node_input_dim, edge_input_dim, output_dim,
                device, num_layer=num_layer, nodes_dim=nodes_dim,
                nedgeoutput=edges_dim, decision_depth=decision_depth,
                final_neuron=final_neuron, readout_type=readout_type,
                level=level, normalize=normalize).to(device)

def run_PNN(h, res_file="./res/test.csv", verbose=True):

    # Vérifier si le test n'a pas déjà été fait
    if tested_before(h, res_file=res_file):
        print("Les hyperparamètres que vous avez fournis ont déjà été testés.")
        return None

    device = h['device']

    # Chargement des données
    train_loader, val_loader, test_loader = load_transform_hot_encod_data(h)

    # Créer une instance du modèle G2N2
    model = generate_model(h)
    model = model.to(device)

    params_number = get_n_params(model)
    print('number of parameters:', params_number)
    h['params_number'] = params_number


    # Lancer l'entraînement et les tests
    perf_results = train_and_test(model, h, train_loader, val_loader, test_loader)

    # Sauvegarder les résultats dans un fichier CSV
    df = save_perf(h, perf_results, file_path=res_file)

    return perf_results


def run_PNN_with_seeds(seed_list):
    i = 1
    for seed_number in seed_list:
        #h = init_hParams()

        seed_everything(seed_number)

        h = dict()

        # Paramètrage du prétraitement des données
        h['normalize']    = True
        h['hot_encoding'] = True

        # Paramètrage de l'optimisation
        h['lr'] = 0.0005
        h['patience'] = 50
        h['step'] = 0.80
        h['epsi'] = 1e-6

        h['l1_reg'] = 0.00    # Coefficient de régularisation L1
        h['l2_reg'] = 0.02      # Coefficient de régularisation L2
        h['weight_decay'] = 0   # Coefficient de régularisation
        h["dropout"] = 0      # paramètre de régularisation

        h['layer_norm'] = True
        h['batch_norm'] = False

        # Paramètrage de la gestion du processus d'entrainement
        h['batch_size'] = 32
        h['ep'] = 1000

        # Paramétrage du modèle de GNN
        h['ntask'] = 0         # Indice de la tâche
        h['depth'] = 5
        h['operator'] = 'adj'  # Opérateur de transformation des données
        h['output_dim'] = 1
        h['num_layer'] = 2
        h["nodes_dim"] = [64]
        h["nodes_dim"] = [h["nodes_dim"][0]] * h["num_layer"]
        h['edges_dim'] = 33
        h['decision_depth'] = 2
        h['final_neuron'] = [128]
        h['readout_type'] = 'sum'
        h['level'] = 'graph'

        # Paramétrage des condition d'exécution
        h['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        h['seed_number'] = seed_number

        print(f"\n\nTest {i}/{len(seed_list)}")
        i += 1
        print(f"Hyper paramètres : \n{h}")

        run_PNN(h, res_file="./res/results_zinc_6Juillet_sum.csv", verbose=False)

list_seeds = [200,300,400,500,600,700,800,900, 1000]
#list_seeds = [150]

run_PNN_with_seeds(list_seeds)
