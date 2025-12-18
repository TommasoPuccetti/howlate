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

    
    def fpr_thr_to_plot(self, df_latency, half, mid_val):
        num_rows = len(df_latency)
        half_index = num_rows // 2
    
        # Select bottom or top half of rows
        if half == 'low_half':
            df_latency = df_latency.iloc[half_index:]  # lower half (bottom)
        elif half == 'high_half':
            df_latency = df_latency.iloc[:half_index]  # upper half (top)
    
        # Select every second row starting from the first
        if mid_val == False:
            df_latency = df_latency.iloc[::2]
    
        return df_latency

    
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
    def plot(self, results_p=None, half='all', mid_val=True, y_scale='linear', attacks='all', name='default'):

        results_p = self.check_if_path_is_given(results_p)
        
        df_latency = pd.read_excel(self.df_results_p, sheet_name='fpr_latency_tradeoff')
        df_sdr = pd.read_excel(self.df_results_p, sheet_name='fpr_sdr_overall')
        
        df_latency.sort_values('target_fpr', inplace=True)
        df_sdr.sort_values('target_fpr', inplace=True)
        
        df_latency = self.fpr_thr_to_plot(df_latency, half, mid_val)
        
        num_points = len(df_latency['target_fpr'])
        x_ticks = np.linspace(0, num_points - 1, num_points)
        plt.figure(figsize=(10, 6), dpi=400)

        if attacks != 'all':
            plt.figure(figsize=(10, 2.1), dpi=400)
            df_latency = df_latency[attacks]
            
        columns_to_plot = [col for col in df_latency.columns if col != "target_fpr"]
            
        cmap = plt.cm.get_cmap('tab20', len(columns_to_plot))
        
        for i, col in enumerate(columns_to_plot):
            plt.plot(x_ticks, df_latency[col], label=col, color=cmap(i))

        plt.subplots_adjust(bottom=0.15)
        plt.xticks(x_ticks, labels=df_latency['target_fpr'])
        
        df_sdr.drop(columns=['target_fpr'], inplace=True)

        ymin, ymax = plt.ylim()
        offset = (ymax - ymin) * 0.04
        
        if attacks != 'all':
            offset = (ymax - ymin) * 0.20
        
        for x, mean_value in zip(x_ticks, df_sdr['sequence_detection_rate']):
            plt.annotate(
                f"{mean_value:.3f}",
                xy=(x, ymin - offset),      # below the axis, in data coords
                xycoords='data',
                ha='center',
                va='top',
                fontsize=9,
                color='red',
                annotation_clip=False       # make sure it’s visible even outside
            )

        
        plt.yscale(y_scale) 
        plt.xlabel('false positive rate ; sequence detection rate', labelpad=15, fontsize=10)
        plt.ylabel('attack latency' + '\n' + '(seconds)', fontsize=10)

        plt.title(self.title, fontsize=14)
        
        if attacks != 'all':
            plt.title(self.title + ' ' + name, fontsize=10)
        
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9.2)
        plt.tight_layout()
        plt.grid(True)
        plt.xticks(fontsize=9.2)
        plt.yticks(fontsize=9.2)
        plt.savefig(os.path.join(results_p, 'final_results_' + '_' + name + '_' + '.pdf'), format='pdf')
        plt.savefig(os.path.join(results_p, 'final_results_' + '_' + name + '_' + '.png'), format='png')
        plt.show()