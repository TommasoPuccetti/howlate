"""

This module evaluates an Intrusion Detection model by computing  standard classification metrics and the time an attack remains undetected. The evaluation includes the following key metrics:

- **False Positive Rate (FPR):** The proportion of normal instances misclassified as attacks.
- **Attack Latency (Δl):** The time taken to detect each attack sequence.
- **Sequence Detection Rate (SDR):** The number of detected attack sequences over the total attack sequences.

### Evaluation Approach
To measure the average latency (ΔL) and SDR at different FPR levels, the evaluation keeps track of:

1. **Initial Data Point of Each Attack Sequence:** Marks the start of each attack.
2. **Positions of Data Points Labeled as Anomalous:** Tracks where the model detects anomalies.
3. **First Correctly Classified Anomalous Data Point in Each Sequence:** Determines when an attack is first detected.

These metrics are only meaningful if the dataset consists of sequences containing normal and anomalous operations.

### Main Evaluation Functions

#### eval_sota()
Evaluates the detector using standard classification metrics, including confusion matrices.
(See implementation: [[evaluator.py#eval_sota]])

#### plot_curves()
Plots precision-recall and ROC curves to visualize model performance.
(See implementation: [[evaluator.py#plot_curves]])

#### eval_fpr_latency()
Computes attack latency for each attack sequence at different FPR thresholds.
(See implementation: [[evaluator.py#eval_fpr_latency]])

#### avg_fpr_latency()
Calculates the average attack latency per attack type and overall.
(See implementation: [[evaluator.py#avg_fpr_latency]])

#### summary_fpr_latency()
Summarizes the average attack latency for different attack types and overall performance.
(See implementation: [[evaluator.py#summary_fpr_latency]])
"""

import os
import numpy as np
import sklearn
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd
from config import *
from loader import PathManager
import warnings
import results_handler as rh
from evaluator import Evaluator

warnings.filterwarnings('ignore')


class SotaEvaluator(Evaluator):
    
    def __init__(self, pm: PathManager):
        self.sota_results_fprs = None
        super().__init__(pm)
     
    def evaluate(self, test_y, test_multi, preds_proba, desired_fprs=DESIRED_FPRS, results_p=None, verbose=False):
        
        results_p = self.check_if_out_path_is_given(results_p)

        self.plot_curves(test_y, preds_proba, results_p=results_p)
        
        attacks = np.unique(test_multi)
        atk_index_list = [np.where(test_multi == value)[0].tolist() for value in attacks]
        
        bin_preds_fpr = self.bin_preds_for_given_fpr(test_y, preds_proba, desired_fprs, verbose)

        sota_results_fprs = []
        
        for bin_pred, desired_fpr in zip(bin_preds_fpr, desired_fprs):
            sota_results_fpr = []

            for indexes, i in zip(atk_index_list, range(0, len(atk_index_list))):

                sota_results = self.eval_sota(test_y[indexes], bin_pred[indexes])
                sota_results['attack'] = attacks[i]
                sota_results_fpr.append(sota_results)
            
            df = pd.DataFrame(sota_results_fpr)
            df.to_csv(os.path.join(results_p, str(desired_fpr) + '.csv'))
            sota_results_fprs.append(df)
            
        self.sota_results_fprs = sota_results_fprs

        return sota_results_fprs

    def evaluate_bin_preds(self, test_y, preds):
        return self.eval_sota(test_y, preds)

    def plot_curves(self, test_y, preds_proba, results_p=None):
            if results_p == None:
                results_p = self.results_p
            self.plot_precision_recall(test_y, preds_proba, results_p=results_p)
            self.plot_roc(test_y, preds_proba, results_p=results_p) 

    def plot_roc(self, test_y, preds_proba, results_p=None):
            if results_p == None:
                results_p = self.results_p
            plt.figure(dpi=400)
            fpr, tpr, _ = metrics.roc_curve(test_y,  preds_proba[:,1])
            plt.plot(fpr, tpr)
            plt.ylabel('Recall', fontsize=20)
            plt.xlabel('False Positive Rate', fontsize=20)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.savefig(os.path.join(results_p, "roc_curve.pdf"), format='pdf', bbox_inches='tight')
            plt.show()

    def plot_precision_recall(self, test_y, preds_proba, results_p=None):
            if results_p == None:
                results_p = self.results_p
            plt.figure(dpi=400)
            fpr, tpr, _ = metrics.precision_recall_curve(test_y,  preds_proba[:,1])
            plt.plot(fpr, tpr)
            plt.ylabel('Precision', fontsize=20)
            plt.xlabel('Recall', fontsize=20)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.savefig(os.path.join(results_p, "precision_recall_curve.pdf"), format='pdf', bbox_inches='tight')
            plt.show()

       




