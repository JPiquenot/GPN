import pandas as pd
import matplotlib.pyplot as plt
import os

def display_resultats(titre: str = "", res_file: str = "per_results.csv", r_score: float = 0.115, type_of_display: str = "last", n: int = 1):
    """
    Display the results of the saved models.

    Parameters:
    - titre (str): The title of the plot.
    - res_file (str): The path to the CSV file containing the results.
    - r_score (float): The reference score to compare with the performance metric.
    - type_of_display (str): The type of models to display ("last", "first", "all").
    - n (int): The number of models to display.
    """

    if not os.path.exists(res_file):
        print(f"Le fichier {res_file} n'existe pas.")
        return

    # Load data from the CSV file
    df = pd.read_csv(res_file)

    # Extract unique model names
    model_names = df['model_name'].unique()

    # Determine the number of models
    num_models = len(model_names)

    # Select the models to display based on the type_of_display parameter
    if type_of_display == "last":
        model_names = model_names[-n:]
    elif type_of_display == "first":
        model_names = model_names[:n]

    for model_name in model_names:
        model_data = df[df['model_name'] == model_name]

        # Extract model parameters
        normalize = model_data['normalize'].values[0]
        layer_norm = model_data['layer_norm'].values[0]
        batch_norm = model_data['batch_norm'].values[0]
        hot_encoding = model_data['hot_encoding'].values[0]
        lr = model_data['lr'].values[0]
        patience = model_data['patience'].values[0]
        ep = model_data['ep'].values[0]
        step = model_data['step'].values[0]
        epsi = model_data['epsi'].values[0]
        batch_size = model_data['batch_size'].values[0]
        depth = model_data['depth'].values[0]
        num_layer = model_data['num_layer'].values[0]
        decision_depth = model_data['decision_depth'].values[0]
        readout_type = model_data['readout_type'].values[0]
        level = model_data['level'].values[0]
        nodes_dim = model_data['nodes_dim'].values[0]
        edges_dim = model_data['edges_dim'].values[0]
        seed_number = model_data['seed_number'].values[0]
        weight_decay = model_data['weight_decay'].values[0]
        dropout = model_data['dropout'].values[0]
        l1_reg = model_data['l1_reg'].values[0]
        l2_reg = model_data['l2_reg'].values[0]
        params_number = model_data['params_number'].values[0]

        # Determine the performance metric and its value
        if "bTest_mae" in model_data.keys():
            perf = model_data['bTest_mae'].min()
            perf_name = "bTest_mae"
        elif "Accuracy" in model_data.keys():
            perf = model_data['Accuracy'].max()
            perf_name = "Accuracy"

        # Create the title for the plot
        title = f"{titre}\n\n"
        title += f'Modèle with {params_number} parameters ({perf_name} = {perf}) \n\n'
        title += f"epochs : {ep}, batch_size : {batch_size}, seed_number : {seed_number}\n"
        title += f"normalize : {normalize}, layer_norm : {layer_norm}, batch_norm : {batch_norm}, hot_encoding : {hot_encoding}\n"
        title += f"num_layer:{num_layer}, nodes_dim: {nodes_dim}, edges_dim : {edges_dim} \n"
        title += f"depth : {depth}, decision_depth : {decision_depth}, readout_type : {readout_type}, level : {level}\n"
        title += f"lr : {lr}, patience : {patience}, step : {step}, epsi : {epsi}\n"
        title += f"dropout : {dropout}, l1_reg : {l1_reg}, l2_reg : {l2_reg}, weight_decay : {weight_decay}\n"

        # Plot the results
        if perf_name == "bTest_mae":
            plot_mae(title, ep, r_score, model_data)
        elif perf_name == "Accuracy":
            plot_accuracy(title, ep, r_score, model_data)

def plot_mae(title: str, ep: int, r_score: float, model_data: pd.DataFrame):
    """
    Plot the results for the mean absolute error (MAE) performance metric.

    Parameters:
    - title (str): The title of the plot.
    - ep (int): The number of epochs.
    - r_score (float): The reference score to compare with the performance metric.
    - model_data (pd.DataFrame): The data for the model.
    """
    epochs = range(1, len(model_data['Train_loss']) + 1)
    reference_score = [r_score for _ in range(ep)]
    plt.plot(epochs, model_data['Train_loss'], label='Train Loss')
    plt.plot(epochs, model_data['Val_loss'], label='Validation Loss')
    plt.plot(epochs, model_data['Test_loss'], label='Test Loss')
    plt.plot(epochs, reference_score, "--", label='score de référence')
    plt.plot(epochs, model_data['bTest_mae'], label='bTest_mae')
    plt.title(title, fontsize=10)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_accuracy(title: str, ep: int, r_score: float, model_data: pd.DataFrame):
    """
    Plot the results for the accuracy performance metric.

    Parameters:
    - title (str): The title of the plot.
    - ep (int): The number of epochs.
    - r_score (float): The reference score to compare with the performance metric.
    - model_data (pd.DataFrame): The data for the model.
    """
    epochs = range(1, len(model_data['Train_loss']) + 1)
    reference_score = [r_score for _ in range(ep)]
    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    f.suptitle(title, fontsize=10)

    ax[0].plot(epochs, model_data['Train_loss'], label='Train Loss')
    ax[0].plot(epochs, model_data['Val_loss'], label='Validation Loss')
    ax[0].plot(epochs, model_data['Test_loss'], label='Test Loss')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].legend()

    ax[1].plot(epochs, reference_score, "--", label='score de référence')
    ax[1].plot(epochs, model_data['Accuracy'], label='Accuracy')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Accuracy')
    ax[1].legend()

    plt.tight_layout()
    plt.show()

display_resultats(res_file="./res/results_zinc.csv", r_score=0.115, type_of_display="last", n=3)
