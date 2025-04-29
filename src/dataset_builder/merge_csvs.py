import os
import pandas as pd
import pickle
import random
from datetime import datetime
import pandas as pd 
import gc
import sys
import os
from pathlib import Path
import numpy as np

def collect_file_paths(input_dir):
    file_paths = [str(p) for p in Path(input_dir).rglob('*') if p.is_file()]
    return file_paths

def save_to_txt(file_paths, output_path):
    with open(output_path, 'w') as f:
        for path in file_paths:
            f.write(f"{path}\n")

def adjust_attack_timestamps_relative(attack_df, timestamp_column, insertion_timestamp):
    attack_df[timestamp_column] = pd.to_datetime(attack_df[timestamp_column], errors='coerce')

    if attack_df[timestamp_column].isnull().all():
        raise ValueError("No valid datetime values in the attack DataFrame.")
    
    time_deltas = attack_df[timestamp_column].diff().fillna(pd.Timedelta(seconds=0))

    new_timestamps = [insertion_timestamp]
    
    for delta in time_deltas[1:]:
        new_timestamps.append(new_timestamps[-1] + delta)
    
    attack_df[timestamp_column] = new_timestamps
    
    return attack_df

def merge(input_folder, files_paths, test_train, feature='None'):
    print(test_train)
    for file in files_paths:
        print(file)
        load_features = pd.read_csv(file, nrows=10).columns
        use_features = list(set(load_features) & set(feature))

        #load attack csv, label, and convert to datetime
        normal_df = pd.read_csv(file, usecols=use_features) 
        for col in feature:
            if col not in normal_df.columns:
                normal_df[col] = pd.NA
        print(normal_df.shape)
        timestamp = 'layers.frame.frame.time'
        normal_df = normal_df.sort_values(by=timestamp)
        file_out =  str(test_train) + "_merged.csv"
        normal_df['layers.frame.frame.time'] = normal_df['layers.frame.frame.time'].str.replace(" CEST", "", regex=False)
        normal_df.to_csv(os.path.join(input_folder, file_out), mode='a', index=False, header=not pd.io.common.file_exists(os.path.join(input_folder, file_out)))

def main():
    input_folder = Path(sys.argv[1])
    features_folder = Path(sys.argv[2])
    train_test = Path(sys.argv[3])

    file_paths = collect_file_paths(input_folder)
    print(file_paths)

    features = np.load(os.path.join(features_folder, "features.npy"))
    
    merge(input_folder, file_paths, train_test, feature=features)
    df = pd.read_csv(os.path.join(input_folder, str(train_test) + "_merged.csv"))
    df['layers.frame.frame.time'] = pd.to_datetime(df['layers.frame.frame.time'], errors='coerce')
    df = df.sort_values(by="layers.frame.frame.time")
    df.to_csv(os.path.join(input_folder, str(train_test) + "_merged.csv"), index=False)
    
if __name__ == "__main__":
    main()