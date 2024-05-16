import matplotlib.pyplot as plt
import os

def plot_and_save_graph(df, column_names, title, xlabel, ylabel, output_filename, output_dir='graph'):
    """
    Affiche et enregistre un graphique des prix de règlement en fonction de la maturité pour une série de dates.

    Parameters:
    df (pd.DataFrame): Le DataFrame contenant les données.
    column_names (list): Liste des colonnes à tracer.
    title (str): Le titre du graphique.
    xlabel (str): Le label de l'axe des x.
    ylabel (str): Le label de l'axe des y.
    output_filename (str): Le nom du fichier de sortie pour enregistrer le graphique.
    output_dir (str): Le répertoire où le graphique sera enregistré. Par défaut, 'graph'.
    """
    plt.figure(figsize=(12, 6))
    for column_name in column_names:
        plt.plot(df.index, df[column_name], label=column_name)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    

    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path)
    
    plt.show()
    print(f"Graph saved to {output_path}")