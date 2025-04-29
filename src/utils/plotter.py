import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import os
from loader import PathManager


class Plotter():
    
    def __init__(self, pm: PathManager, title):
        self.results_p = pm.results_p
        self.title = title
        self.df_results_p = os.path.join(self.results_p, 'final_results.xlsx')

    def check_if_path_is_given(self, path):
        #if the path is not provided by argument take the one in object param.
        if path == None:
            path = self.results_p
        else:
            self.df_results_p = os.path.join(results_p, 'final/final_results.xlsx')
        
        return path

    def plot(self, results_p=None):

        results_p = self.check_if_path_is_given(results_p)
        
        df_latency = pd.read_excel(self.df_results_p, sheet_name='fpr_latency_tradeoff')
        df_sdr = pd.read_excel(self.df_results_p, sheet_name='fpr_sdr_overall')
        
        df_latency.sort_values('target_fpr', inplace=True)
        df_sdr.sort_values('target_fpr', inplace=True)

        num_points = len(df_latency['target_fpr'])
        x_ticks = np.linspace(0, num_points - 1, num_points)

        plt.figure(figsize=(6, 4), dpi=400)
        
        # Exclude the target column (replace 'fpr' with the actual column name if different)
        columns_to_plot = [col for col in df_latency.columns if col != "target_fpr"]

        # Plot each selected column
        for col in columns_to_plot:
            plt.plot(x_ticks, df_latency[col], label=col)

        plt.subplots_adjust(bottom=0.15)
        # Set x-ticks to be the normalized values
        plt.xticks(x_ticks, labels=df_latency['target_fpr'])
        
        
        # Compute mean for each row (excluding 'target_fpr' column)
        df_sdr.drop(columns=['target_fpr'], inplace=True)
    
        # Annotate with mean values
        for i, (x, mean_value) in enumerate(zip(x_ticks, df_sdr['percent_detected_sequences'])):
            plt.annotate(f"{mean_value:.3f}", (x, 0), 
                         textcoords="offset points", xytext=(0, -35), 
                         ha='center', fontsize=8, color='red')
        
        plt.xlabel('false positive rate ; sequence detection rate', labelpad=15, fontsize=12)
        plt.ylabel('attack latency (seconds)', fontsize=12)
        plt.title(self.title, fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.savefig(os.path.join(results_p, 'final_results.pdf'), format='pdf')
        plt.show()