import matplotlib.pyplot as plt
import pandas as pd


def plot_time_series(df, column_name,title,  xlabel, ylabel):
    plt.figure(figsize=(10, 6))  
    plt.plot(df['index'], df[column_name])  
    plt.title(title)  
    plt.xlabel(xlabel)  
    plt.ylabel(ylabel) 
    plt.legend() 
    plt.grid(True)
    plt.ylim(bottom=0)
    plt.xlim(left=df['index'].min(), right=df['index'].max())
    plt.show() 


