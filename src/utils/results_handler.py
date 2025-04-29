import pandas as pd
from config import *
import os 

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
    """
    TODO verbose
    print("Accuracy: {}, \nRecall: {}, \nPrecision: {}, \nF1-Score: {}".format(acc, rec, prec, f1))
    print("TP: {}, \nFP: {}, \nTN: {}, \nFN: {}".format(tp, fp, tn, fn))
    print(PRINT_SEPARATOR)
    """

    return sota_results

def init_sequence_results_dict():
    return pd.DataFrame(columns=[
        'start_idx_attack', 'end_idx_attack', 'attack_duration', 'time_to_detect',
        'idx_detection_abs', 'idx_detection_rel', 'attack_len', 'attack_type',
        'pr', 'rec', 'fpr', 'tn', 'fp', 'fn', 'tp', 'target_fpr', 'detected'])

def store_sequence_results(df, latency_seq_res, seq_sota_eval, y_test_atk, test_multi, desired_fpr):
    """
    Stores attack detection results into a DataFrame and returns the updated DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to store results.
        latency_seq_res (dict): Latency sequence results.
        seq_sota_eval (dict): Evaluation metrics.
        y_test_atk (list): Attack labels for the test set.
        test_multi (list): Attack types.
        desired_fpr (float): Target false positive rate.

    Returns:
        pd.DataFrame: Updated DataFrame with stored results.
    """
    
    new_row = {
        'start_idx_attack': latency_seq_res['atk_start_idx'],
        'end_idx_attack': latency_seq_res['atk_end_idx'],
        'attack_duration': latency_seq_res['atk_time'],
        'time_to_detect': latency_seq_res['det_time'],
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

def store_results_for_attack_type(df):
    # Calculate average, min, max, and std for each group
    avg_result_df = df.agg({
        'attack_len': ['mean', 'min', 'max', 'std'],
        'fpr': 'mean',
        'pr': 'mean',
        'rec': 'mean',
        'time_to_detect': ['mean', 'min', 'max', 'std'],
        'idx_detection_rel': ['mean', 'min', 'max', 'std']}).reset_index()
    avg_result_df.columns = ['_'.join(col) for col in avg_result_df.columns]
    avg_result_df = convert_timedelta_to_seconds(avg_result_df)
    return avg_result_df

def store_overall_results(target_fpr, df, num_detected):
    # Calculate aggregated statistics
    result_dict = {
        'detected_sequences': num_detected,
        'all_sequences': df.shape[0],
        'percent_detected_sequences': num_detected / df.shape[0],
        'avg_time_to_detect': df['time_to_detect'].mean(),
        'std_time_to_detect': df['time_to_detect'].std(),
        'min_time_to_detect': df['time_to_detect'].min(),
        'max_time_to_detect': df['time_to_detect'].max(),
        'avg_idx_detection_rel': df['idx_detection_rel'].mean(),
        'std_idx_detection_rel': df['idx_detection_rel'].std(),
        'min_idx_detection_rel': df['idx_detection_rel'].min(),
        'max_idx_detection_rel': df['idx_detection_rel'].max(),
        'target_fpr': target_fpr}
    all_results_df = pd.DataFrame([result_dict])
    
    return all_results_df

def convert_timedelta_to_seconds(data):
        for col in data.select_dtypes(include=['timedelta64']):
            data[col] = data[col].dt.total_seconds()  # Convert to float seconds
        return data

def all_latency_results_to_excel(results_p, target_fpr, df, avg_result_df, detection_rate_df, all_results_df):

    df = convert_timedelta_to_seconds(df)
    df_detect = df[df['detected'] != 0]
    
    # Save the result and another DataFrame in the same Excel file
    with pd.ExcelWriter(os.path.join(results_p,  target_fpr + '_all.xlsx'), engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='all_sequences_results')
        df_detect.to_excel(writer, index=False, sheet_name='detected_sequences_results')
        avg_result_df.to_excel(writer, index=False, sheet_name='avg_results_for_attack_type')
        detection_rate_df.to_excel(writer, index=False, sheet_name='detection_rate_for_attack_type')
        all_results_df.to_excel(writer, index=False, sheet_name='overall_results')
        all_results_df[['detected_sequences', 'all_sequences', 'percent_detected_sequences', 'target_fpr']].to_excel(writer, index=False, sheet_name='detection_rate_overall')

def summary_fpr_latency_sdr_to_excel(results_p, df_fpr_out, df_sdr_out, df_sdr_out_all):
    # Save the result and another DataFrame in the same Excel file
    with pd.ExcelWriter(os.path.join(results_p,  'final_results.xlsx'), engine='xlsxwriter') as writer:
        df_fpr_out.to_excel(writer, index=False, sheet_name='fpr_latency_tradeoff')
        df_sdr_out.to_excel(writer, index=False, sheet_name='fpr_sdr_tradeoff')
        df_sdr_out_all.to_excel(writer, index=False, sheet_name='fpr_sdr_overall')

def create_folder_if_not_exist(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)