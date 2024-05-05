import matplotlib.pyplot as plt
import pandas as pd


def plot_time_series(df, column_names, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    for column_name in column_names:
        plt.plot(df.index, df[column_name], label=column_name)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
