import pandas as pd
from src.utils.config import *
import os 

"""
This module handles the aggregation and export of evaluation results for an Intrusion Detection model.

### Result Handling Overview

Steps:

1. **Per-Sequence Results:** Captures detection time, latency, and classification metrics per attack sequence.
2. **Aggregate Results by Attack Type:** Computes average and statistical summaries (mean, min, max, std) of performance per attack type.
3. **Overall Evaluation Metrics:** Summarizes results across all sequences.
4. **Exporting Results to Excel:** Consolidates detailed and summary outputs into structured Excel files with multiple sheets.

### Main Result Handling Functions

#### store_sequence_results()
Appends detection results for a single attack sequence to a DataFrame
(See implementation: [here](results_handler.html#store_sequence_results))

#### store_results_for_attack_type()
Aggregates per-sequence results into statistical summaries grouped by attack type.
(See implementation: [here](results_handler.html#store_results_for_attack_type))

#### store_overall_results()
Computes overall detection performance metrics across all sequences.
(See implementation: [here](results_handler.html#store_overall_results))

#### convert_timedelta_to_seconds()
Converts timedelta fields in result DataFrames to float seconds.
(See implementation: [here](results_handler.html#convert_timedelta_to_seconds))

#### all_latency_results_to_excel()
Exports detailed per-sequence results and summary statistics to an Excel file with multiple sheets.
(See implementation: [here](results_handler.html#all_latency_results_to_excel))

#### summary_fpr_latency_sdr_to_excel()
Exports summary tables of latency and SDR tradeoffs across FPR thresholds to Excel file.
(See implementation: [here](results_handler.html#summary_fpr_latency_sdr_to_excel))
"""

def store_sota_results(acc, rec, prec, f1, fpr, tn, fp, fn, tp):
    sota_results = {
        "accuracy": acc,
        "recall": rec,
        "precision": prec,
        "f1-score": f1,
        "fpr": fpr,
        "tn": tn,
        "fp":fp,
        "fn": fn,
        "tp": tp
    }

    return sota_results

def init_sequence_results_dict():
    return pd.DataFrame(columns=[
        'start_idx_attack', 'end_idx_attack', 'attack_duration', 'attack_latency',
        'idx_detection_abs', 'idx_detection_rel', 'attack_len', 'attack_type',
        'pr', 'rec', 'fpr', 'tn', 'fp', 'fn', 'tp', 'target_fpr', 'detected'])

# === store_sequence_results ===
    
"""
This function appends detection results for a single attack sequence to an evaluation DataFrame.
It gathers latency, classification, and sequence metadata into a structured format for further 
analysis or reporting, especially when evaluating classifier performance over sequences.
    
Parameters:
    
- **df:** Existing pandas DataFrame collecting results across sequences.
- **latency_seq_res:** Dictionary containing detection latency and index information for the current sequence.
- **seq_sota_eval:** Dictionary with classification metrics (e.g., precision, recall, FPR) for the sequence.
- **y_test_atk:** Ground truth binary labels for the attack sequence (typically all 1s).
- **test_multi:** Array of multi-label annotations (e.g., attack types), indexed the same as the test set.
- **desired_fpr:** The specific false positive rate threshold being evaluated.
    
Returns:
    
- **df:** Updated pandas DataFrame with a new row containing the evaluation results for the current sequence.
"""
def store_sequence_results(df, latency_seq_res, seq_sota_eval, y_test_atk, test_multi, desired_fpr):
    #print("DEBUG - Esempio test_multi:", test_multi[:10])
    #print("DEBUG - Tipo elemento test_multi:", type(test_multi[0]))
    new_row = {
        'start_idx_attack': latency_seq_res['atk_start_idx'],
        'end_idx_attack': latency_seq_res['atk_end_idx'],
        'attack_duration': latency_seq_res['atk_time'],
        'attack_latency': latency_seq_res['det_time'],
        'idx_detection_abs': latency_seq_res['det_idx_abs'],
        'idx_detection_rel': latency_seq_res['det_idx_rel'],
        'attack_len': len(y_test_atk),
        'attack_type': test_multi[latency_seq_res['atk_start_idx']],
        'pr': seq_sota_eval['precision'],
        'rec': seq_sota_eval['recall'],
        'fpr': seq_sota_eval['fpr'],
        'fp': seq_sota_eval['fp'],
        'fn': seq_sota_eval['fn'],
        'tp': seq_sota_eval['tp'],
        'tn': seq_sota_eval['tn'],
        'target_fpr': desired_fpr,
        'detected': latency_seq_res['det']
    }
    
    return pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

# === store_results_for_attack_type ===

"""
This function aggregates sequence-level evaluation results for a specific attack type.
It computes descriptive statistics (mean, min, max, std) for key performance metrics 
such as attack duration, latency, detection index, precision, recall, and FPR.
    
Parameters:
    
- **df:** DataFrame containing per-sequence evaluation results, typically generated using 
      `store_sequence_results`.
    
Returns:
    
 - **avg_result_df:** DataFrame with aggregated statistics across sequences, with time-based 
    fields (e.g., latency) converted to seconds for interpretability.
"""
def store_results_for_attack_type(df):
        
    avg_result_df = df.agg({
        'attack_len': ['mean', 'min', 'max', 'std'],
        'fpr': 'mean',
        'pr': 'mean',
        'rec': 'mean',
        'attack_latency': ['mean', 'min', 'max', 'std'],
        'idx_detection_rel': ['mean', 'min', 'max', 'std']}).reset_index()
    avg_result_df.columns = ['_'.join(col) for col in avg_result_df.columns]
    avg_result_df = convert_timedelta_to_seconds(avg_result_df)
    
    return avg_result_df

    
# === store_overall_results ===

"""
This function summarizes overall detection performance across all attack sequences for a given FPR.
It computes aggregate metrics such as sequence detection rate, latency statistics, and relative 
detection index statistics, encapsulating them in a single-row DataFrame for concise reporting.
    
Parameters:
    
- **target_fpr:** The false positive rate threshold under evaluation.
- **df:** DataFrame containing per-sequence evaluation results (from `store_sequence_results`).
- **num_detected:** Number of sequences where an attack was successfully detected.
    
Returns:
    
- **all_results_df:** Single-row DataFrame containing summary statistics for all sequences at the given FPR.
"""
def store_overall_results(target_fpr, df, num_detected):

    result_dict = {
        'detected_sequences': num_detected,
        'all_sequences': df.shape[0],
        'sequence_detection_rate': num_detected / df.shape[0],
        'avg_attack_latency': df['attack_latency'].mean(),
        'std_attack_latency': df['attack_latency'].std(),
        'min_attack_latency': df['attack_latency'].min(),
        'max_attack_latency': df['attack_latency'].max(),
        'avg_idx_detection_rel': df['idx_detection_rel'].mean(),
        'std_idx_detection_rel': df['idx_detection_rel'].std(),
        'min_idx_detection_rel': df['idx_detection_rel'].min(),
        'max_idx_detection_rel': df['idx_detection_rel'].max(),
        'target_fpr': target_fpr}
    all_results_df = pd.DataFrame([result_dict])
    
    return all_results_df

# === convert_timedelta_to_seconds ===

"""
This utility function converts all timedelta columns in a DataFrame to floating-point 
values representing total seconds. It is useful for normalizing time-based metrics 
into a consistent and interpretable format.
    
Parameters:
    
- **data:** pandas DataFrame potentially containing columns of dtype 'timedelta64'.
    
Returns:
    
- **data:** The same DataFrame with all timedelta columns converted to float seconds.
"""
def convert_timedelta_to_seconds(data):

    for col in data.select_dtypes(include=['timedelta64']):
            data[col] = data[col].dt.total_seconds()  # Convert to float seconds
    return data


# === all_latency_results_to_excel ===

"""
This function exports detailed and summary detection results to an Excel file with multiple sheets.
It converts all time-based columns to seconds, filters detected sequences, and saves comprehensive
results, averages, and detection statistics for the evaluated FPR.

Parameters:

- **results_p:** Path to the directory where the Excel file should be saved.
- **target_fpr:** String label for the current false positive rate (used in the filename).
- **df:** DataFrame with per-sequence evaluation results.
- **avg_result_df:** DataFrame with aggregated statistics for each attack type.
- **detection_rate_df:** DataFrame with detection rates grouped by attack type.
- **all_results_df:** DataFrame containing overall summary metrics across all sequences.

Returns:

- **None** — results are written directly to an Excel file at the specified location.
"""
def all_latency_results_to_excel(results_p, target_fpr, df, avg_result_df, detection_rate_df, all_results_df):

    df = convert_timedelta_to_seconds(df)
    df_detect = df[df['detected'] != 0]
    
    with pd.ExcelWriter(os.path.join(results_p,  target_fpr + '_lat.xlsx'), engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='all_sequences_results')
        df_detect.to_excel(writer, index=False, sheet_name='detected_sequences_results')
        avg_result_df.to_excel(writer, index=False, sheet_name='avg_results_for_attack_type')
        detection_rate_df.to_excel(writer, index=False, sheet_name='detection_rate_for_attack_type')
        all_results_df.to_excel(writer, index=False, sheet_name='overall_results')
        all_results_df[['detected_sequences', 'all_sequences', 'sequence_detection_rate', 'target_fpr']].to_excel(writer, index=False, sheet_name='detection_rate_overall')

# === summary_fpr_latency_sdr_to_excel ===

"""
This function exports summary evaluation results across different FPRs to a single Excel file.
It organizes data into separate sheets for latency trade-offs, per-type sequence detection rates (SDR),
and overall SDR statistics, enabling a clear comparison of model performance across thresholds.

Parameters:

- **results_p:** Path to the directory where the Excel file should be saved.
- **df_fpr_out:** DataFrame containing FPR vs. latency trade-off data.
- **df_sdr_out:** DataFrame with sequence detection rate (SDR) by attack type for each FPR.
- **df_sdr_out_all:** DataFrame with overall SDR statistics across all attack types and FPRs.

Returns:

- **None** — results are written directly to an Excel file named `final_results.xlsx`.
"""

def summary_fpr_latency_sdr_to_excel(results_p, df_fpr_out, df_sdr_out, df_sdr_out_all):
    with pd.ExcelWriter(os.path.join(results_p,  'final_results.xlsx'), engine='xlsxwriter') as writer:
        df_fpr_out.to_excel(writer, index=False, sheet_name='fpr_latency_tradeoff')
        df_sdr_out.to_excel(writer, index=False, sheet_name='fpr_sdr_tradeoff')
        df_sdr_out_all.to_excel(writer, index=False, sheet_name='fpr_sdr_overall')

def create_folder_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)