"""
This module define Plotter class that plots FPR / Latency tradeoff.
"""

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

    # === check_if_path_is_given ===

    """
    Utility method to determine the base output path and update internal results file path.
    
    Parameters:
    
    - path: Optional path provided externally.
    
    Returns:
    
    - The provided path if given; otherwise, returns the default object path (`self.results_p`).
    
    Side effects:
    
    - If an explicit path is given, sets `self.df_results_p` to point to the expected Excel file at `[path]/final/final_results.xlsx`.
    
    Notes:
    
    - Used primarily to ensure consistent access to the summary Excel file for plotting or evaluation.
    """
    def check_if_path_is_given(self, path):
        if path == None:
            path = self.results_p
        else:
            self.df_results_p = os.path.join(results_p, 'final/final_results.xlsx')
        
        return path

    # === plot ===

    """
    Plots the tradeoff between attack detection latency and false positive rate (FPR),
    annotated with the sequence detection rate (SDR) at each FPR level.
    
    Parameters:
    
    - results_p: Optional path to save the plot. If None, defaults to the internal path.
    
    Behavior:
    
    - Loads two Excel sheets:
        - 'fpr_latency_tradeoff': Contains latency values per FPR.
        - 'fpr_sdr_overall': Contains SDR statistics per FPR.
    - Sorts both dataframes by FPR and prepares a plot showing each latency metric.
    - Annotates the plot with the SDR (percent of detected sequences) below each FPR tick.
    
    Returns:
    
    - None. Saves the plot as `final_results.pdf` and displays it.
    
    Notes:
    
    - Designed to visually assess how changes in FPR affect detection latency and overall performance.
    """
    def plot(self, results_p=None):

        results_p = self.check_if_path_is_given(results_p)
        
        df_latency = pd.read_excel(self.df_results_p, sheet_name='fpr_latency_tradeoff')
        df_sdr = pd.read_excel(self.df_results_p, sheet_name='fpr_sdr_overall')
        
        df_latency.sort_values('target_fpr', inplace=True)

        """
         # === NORMALIZZAZIONE DEI VALORI DI LATENZA ===
        # Normalizza tutti i valori tranne 'target_fpr' tra 0 e 1
        for col in df_latency.columns:
            if col != 'target_fpr':
                min_val = df_latency[col].min()
                max_val = df_latency[col].max()
                if max_val != min_val:
                    df_latency[col] = (df_latency[col] - min_val) / (max_val - min_val)
        """









        df_sdr.sort_values('target_fpr', inplace=True)

        num_points = len(df_latency['target_fpr'])
        x_ticks = np.linspace(0, num_points - 1, num_points)

        plt.figure(figsize=(6, 4), dpi=400)
        
        columns_to_plot = [col for col in df_latency.columns if col != "target_fpr"]

        for col in columns_to_plot:
            plt.plot(x_ticks, df_latency[col], label=col)

        plt.subplots_adjust(bottom=0.15)
        plt.xticks(x_ticks, labels=df_latency['target_fpr'])
        
        df_sdr.drop(columns=['target_fpr'], inplace=True)
    
        for i, (x, mean_value) in enumerate(zip(x_ticks, df_sdr['sequence_detection_rate'])):
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