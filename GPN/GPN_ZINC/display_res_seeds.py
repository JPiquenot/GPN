import pandas as pd
import matplotlib.pyplot as plt

def display_res_seeds(res_file):
    df_res = pd.read_csv(res_file)
    df_res_grouped = df_res.groupby('seed_number')['bTest_mae'].min().reset_index()
    df_res_grouped.set_index("seed_number", inplace=True)
    
    mean = df_res_grouped.bTest_mae.mean()
    max = df_res_grouped.bTest_mae.max()
    std = df_res_grouped.bTest_mae.std()
    
    hparams = df_res.iloc[0,4:-2]
    hparams = hparams.drop('seed_number')
    print("Résultats Test PNN sur le dataset ZINC\n")
    print(f"Voici les hyperparamètres : \n\n{hparams}")
    
    titre = "\n\nRésultats des tests PNN sur le dataset ZINC"
    titre = titre + "\nbest_MAE = {:.4f} ∓ {:.4f}".format(mean, std)
    df_res_grouped.plot(kind="bar", title=titre)
    
    plt.show()

res_file = "./res/results_zinc_6Juillet_sum.csv"
display_res_seeds(res_file)
