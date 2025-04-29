"""
This module evaluates an Intrusion Detection model by computing the time an attack remains undetected. The evaluation includes the following key metrics:

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

#### evaluate()
Evaluates the detector using latency/fpr tradeoff. It computes latency, for each of the attack sequences in the dataset at different FPR thresholds. Implements the abstract methods of the Evaluator class in [evaluator.py](evaluator.html). Main calls are [avg_fpr_latency()](evaluator_latency.html#avg_fpr_latency) and [summary_fpr_latency()](evaluator_latency.html#summary_fpr_latency)  functions.  
(See implementation: [here](evaluator_latency.html#evaluate))

#### avg_results() 
Computes the average latency results at various FPR thresholds for sequences, both overall and grouped by attack type. The output is a .xlsx file resuming collecting results for each attack sequence, and detected attack sequence, along with the average and overall results for latency and SDR.  
(See implementation: [here](evaluator_latency.html#avg_fpr_latency)).

#### summary_fpr_latency()
Create a table that reports the average results for latency and SDR at different FPR thresholds.
(See implementation: [here](evaluator_latency.html#summary_fpr_latency))

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


class LatencyEvaluator(Evaluator):
    
    def __init__(self, results_p: PathManager):
        self.avg_results = None
        self.tradeoff_summary = None
        super().__init__(results_p)
    
    # === evaluate ===
    def evaluate(self, test_y, test_multi, test_timestamp, test_seq, preds_proba, desired_fprs=DESIRED_FPRS, results_p=None, verbose=False):
        
        results_p = self.check_if_out_path_is_given(results_p)

        bin_preds_fpr = self.bin_preds_for_given_fpr(test_y, preds_proba, desired_fprs, verbose)

        sequences_results_fprs = []
        for bin_pred, des_fpr in zip(bin_preds_fpr, desired_fprs):
            sequences_results = self.eval_all_attack_sequences(test_y, test_multi, test_timestamp, test_seq, bin_pred, des_fpr, results_p, verbose)
            sequences_results_fprs.append(sequences_results)

        self.avg_results = self.avg_fpr_latency(sequences_results_fprs)
        self.tradeoff_summary =  self.summary_fpr_latency()

        return self.avg_results, self.tradeoff_summary
        
    # === atk_sequence_from_seq_idxs ===
    def atk_sequence_from_seq_idxs(self, test_y, bin_pred, seq, last):
        seq_y = test_y[seq]
        seq_preds = np.array(bin_pred[last: last + seq_y.shape[0]])
        y_test_atk = np.where(seq_y == 1)[0]
        if len(y_test_atk) == 0:
            last += len(seq_y)

        return seq_y, seq_preds, y_test_atk, last

    # === eval_sequence_latency ===
    def eval_sequence_latency(self, seq, y_test_atk, test_timestamp, seq_preds):
        # Compute attack timing
        attack_start_idx = seq[y_test_atk[0]]
        attack_end_idx = seq[y_test_atk[-1]]
        attack_time = test_timestamp[attack_end_idx] - test_timestamp[attack_start_idx]
        
        # Detect first attack occurrence
        if 1 in seq_preds[y_test_atk]:
            index_rel = np.where(seq_preds[y_test_atk] == 1)[0][0]
            index_abs = seq[y_test_atk[index_rel]]
            detection_time = test_timestamp[index_abs] - test_timestamp[attack_start_idx]
            detected = 1
        else:
            index_rel = len(seq) - 1
            index_abs = seq[y_test_atk[index_rel]]
            detection_time = attack_time  # If undetected, assign full attack time
            detected = 0

        latency_seq_res = {
            "atk_start_idx": attack_start_idx,
            "atk_end_idx": attack_end_idx,
            "atk_time": attack_time,
            "det_idx_rel": index_rel,
            "det_idx_abs": index_abs,
            "det_time": detection_time,
            "det": detected
        }
        
        return latency_seq_res

    # === eval_all_attack_sequences ===
    def eval_all_attack_sequences(self, test_y, test_multi, test_timestamp, test_seq, bin_pred, desired_fpr, results_p, verbose):
        sequences_results = rh.init_sequence_results_dict()
        last = 0  
        for i, seq in enumerate(test_seq):
            seq_y, seq_preds, y_test_atk, last = self.atk_sequence_from_seq_idxs(test_y, bin_pred, seq, last)
            seq_sota_eval = self.eval_sota(seq_y, seq_preds)
            latency_seq_res = self.eval_sequence_latency(seq, y_test_atk, test_timestamp, seq_preds)
            sequences_results = rh.store_sequence_results(sequences_results, latency_seq_res, seq_sota_eval, y_test_atk, test_multi, desired_fpr)
            last += len(seq_y)
        if verbose: 
            sequences_results.to_csv(os.path.join(results_p,  str(desired_fpr) + 'verb.csv'), index=None)
        return sequences_results

    # === avg_fpr_latency ===
    def avg_fpr_latency(self, sequences_results, results_p=None):
        #if the path is not provided by argument take the one in object param.
        results_p = self.check_if_out_path_is_given(results_p)
            
        for df in sequences_results: 
            num_seq = df.shape[0]
            # calculate time_to_detect (attack latency) for all the detected sequences 
            df['time_to_detect'] = pd.to_timedelta(df['time_to_detect']).dt.total_seconds()
            df_detect = df[df['detected'] != 0]
            #calculate sequence detection rate
            grouped_df = df.groupby('attack_type')
            grouped_df_det = df_detect.groupby('attack_type').size().reset_index(name='count_det')
            grouped_df_tot = df.groupby('attack_type').size().reset_index(name='count_tot') 
            
            detection_rate_df = pd.merge(grouped_df_det, grouped_df_tot, on='attack_type', how='outer')
            detection_rate_df['count_ratio'] = detection_rate_df['count_det'] / detection_rate_df['count_tot']
            target_fpr = str(df['target_fpr'].unique()[0])
            detection_rate_df['target_fpr'] = target_fpr
            avg_result_df = rh.store_results_for_attack_type(grouped_df)
            all_results_df = rh.store_overall_results(target_fpr, df, df_detect.shape[0])
            rh.all_latency_results_to_excel(results_p, target_fpr, df, avg_result_df, detection_rate_df, all_results_df)

    # === summary_fpr_latency ===
    def summary_fpr_latency(self, results_p=None):
        results_p = self.check_if_out_path_is_given(results_p)
        files = os.listdir(results_p)
        xlsx_files = [file for file in files if file.endswith('.xlsx')]
    
        df_out = pd.DataFrame()
        rows_fpr = []
        rows_sdr = []
        rows_sdr_all = []
        
        for file in xlsx_files:
            df_fpr = pd.read_excel(os.path.join(results_p, file) , sheet_name='avg_results_for_attack_type')
            df_sdr = pd.read_excel(os.path.join(results_p, file) , sheet_name='detection_rate_for_attack_type')
            df_sdr_all = pd.read_excel(os.path.join(results_p, file) , sheet_name='detection_rate_overall')
            
            target_fpr = df_sdr['target_fpr'].unique()[0]
            
            df_fpr_out = df_fpr.set_index('attack_type_').T
            selected_row = df_fpr_out.loc['time_to_detect_mean']
            selected_row = selected_row.to_frame().T
            selected_row['target_fpr'] = [target_fpr]
            rows_fpr.append(selected_row)

            df_sdr_out = df_sdr.set_index('attack_type').T
            selected_row = df_sdr_out.loc['count_ratio']
            selected_row = selected_row.to_frame().T
            selected_row['target_fpr'] = [target_fpr]
            rows_sdr.append(selected_row)
            rows_sdr_all.append(df_sdr_all)

            
        df_fpr_out = pd.concat(rows_fpr, ignore_index=True)
        df_sdr_out = pd.concat(rows_sdr, ignore_index=True)
        df_sdr_out_all = pd.concat(rows_sdr_all, ignore_index=True)
        
        rh.summary_fpr_latency_sdr_to_excel(results_p, df_fpr_out, df_sdr_out, df_sdr_out_all)

        return (df_fpr_out, df_sdr_out)



