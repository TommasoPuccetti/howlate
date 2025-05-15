import numpy as np
import os
import pandas as pd
import pickle
import random
from collections import Counter
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import re
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent / "../utils"))
from loader  import PathManager

class DatasetBuilder():
    
    def __init__(self, pm: PathManager):
        self.labels = pm.labels
        self.ntest_csv_p = pm.ntest_csv_p
        self.ntrain_csv_p = pm.ntrain_csv_p
        self.atest_csvs_p = pm.atest_csvs_p
        self.atrain_csvs_p = pm.atrain_csvs_p
        self.ntest_csvs_p = pm.ntest_csvs_p
        self.ntrain_csvs_p = pm.ntrain_csvs_p
        self.train_test = pm.train_test
        self.out_train_p = pm.out_train_p
        self.out_test_p = pm.out_test_p
        self.files = pm.files

    def select_insertion_point(self, df, timestamp_column, last_timestamp):
    
        start_time = last_timestamp
        end_time = start_time +  pd.Timedelta(minutes=random.uniform(5, 15))
    
        from_last_data = df[(df[timestamp_column] >= start_time) & (df[timestamp_column] < end_time)]
        
        if not from_last_data.empty:
            selected_row = from_last_data.sample(n=1)
            selected_index = selected_row.index[0]
            selected_timestamp = selected_row[timestamp_column].iloc[0]
            return selected_index, selected_timestamp
        else:
            raise ValueError("No data points found within the first hour.")
            
    def adjust_attack_timestamps_relative(self, attack_df, timestamp_column, insertion_timestamp):
        
        attack_df[timestamp_column] = pd.to_datetime(attack_df[timestamp_column], errors='coerce')
    
        if attack_df[timestamp_column].isnull().all():
            raise ValueError("No valid datetime values in the attack DataFrame.")
        
        time_deltas = attack_df[timestamp_column].diff().fillna(pd.Timedelta(seconds=0))
        new_timestamps = [insertion_timestamp]
        
        for delta in time_deltas[1:]:
            new_timestamps.append(new_timestamps[-1] + delta)
        
        attack_df[timestamp_column] = new_timestamps
        
        return attack_df, attack_df['layers.frame.frame.time'].max()

    def merge(self, normal_df, attack_df, timestamp_column, features):
    
        normal_df.reset_index(drop=True, inplace=True)
        attack_df.reset_index(drop=True, inplace=True)
        selected_columns = [col for col in features if col in attack_df.columns]
        for col in features:
            if col not in normal_df.columns:
                normal_df[col] = pd.NA
            if col not in attack_df.columns:
                attack_df[col] = pd.NA
                
        normal_df = normal_df[features]
        attack_df = attack_df[features]
        
        merged_df = pd.concat([normal_df, attack_df[selected_columns]], ignore_index=True)
        merged_df = merged_df.sort_values(by=timestamp_column, ascending=True)
        merged_df.reset_index(drop=True, inplace=True)
    
        return merged_df
        
    def create_attack_sequences(self, test_train, feature='None'):

        n_path = self.ntest_csv_p
        a_paths = self.atest_csvs_p
        o_path = self.out_test_p
        if test_train == 'train':
            n_path = self.ntrain_csv_p
            a_paths = self.atrain_csvs_p
            o_path = self.out_train_p
        
        normal_df = pd.read_csv(n_path)
        features = normal_df.columns
        features = features.to_list()
        features.append('label')
        features.append('sequence')
        
        timestamp = 'layers.frame.frame.time'
        normal_df['label'] = 'normal'
        normal_df[timestamp] = normal_df[timestamp].str.replace(" CEST", "", regex=False)
        normal_df[timestamp] = pd.to_datetime(normal_df[timestamp], errors='coerce')
        
        last_timestamp = normal_df[timestamp].min()
        
        for i, file in enumerate(a_paths):
            print(i)
            print(file)
            attack_df = pd.read_csv(file, nrows=10)
            selected_columns = [col for col in features if col in attack_df.columns]
            print(len(selected_columns))
            attack_df = pd.read_csv(file, usecols=selected_columns) 
            for col in features:
                if col not in attack_df.columns:
                    attack_df[col] = pd.NA
            attack_df = attack_df[features]
            print(attack_df.shape)  
            attack_df[timestamp] = attack_df[timestamp].str.replace(" CEST", "", regex=False)
            attack_df[timestamp] = pd.to_datetime(attack_df[timestamp], errors='coerce')
        
            # shorten attack csv to have a sequence of 30 to 180 seconds
            end_time_1 = self.shorten_attack_duration(attack_df, timestamp, 0)
            end_time_2 = self.shorten_attack_duration(attack_df, timestamp, end_time_1)
            attack_df_1 = attack_df[attack_df[timestamp] <= end_time_1]
            attack_df_2 = attack_df[(attack_df[timestamp] > end_time_1) & (attack_df[timestamp] <= end_time_2)]
            attack_df_1['label'] = file[:-6]
            attack_df_1['sequence'] = i
            attack_df_2['label'] = file[:-6]
            attack_df_2['sequence'] = i + 0.5
            print(attack_df.shape)
        
            #select a timestamp to merge attack sequence in normal and save last attack package
            insert_index, insert_timestamp = self.select_insertion_point(normal_df, timestamp, last_timestamp)
            
            old_last_timestamp = last_timestamp 
        
            print(old_last_timestamp)
            print(insert_timestamp)
            
            #merge and return last_timestamp of sequence
            attack_df_1, last_timestamp = self.adjust_attack_timestamps_relative(attack_df_1, timestamp, insert_timestamp)
            #print(last_timestamp)
            normal_df_temp = normal_df[(normal_df[timestamp] > old_last_timestamp) & (normal_df[timestamp] < last_timestamp)]
            
            merged_df = self.merge(normal_df_temp, attack_df_1, timestamp, features)
            
            merged_df[features].to_csv(o_path, mode='a', index=False, header=not pd.io.common.file_exists(o_path))
        
            #select a timestamp to merge attack sequence in normal and save last attack package
            insert_index, insert_timestamp = self.select_insertion_point(normal_df, timestamp, last_timestamp)
        
            old_last_timestamp = last_timestamp 

            #merge and return last_timestamp of sequence
            attack_df_2, last_timestamp = self.adjust_attack_timestamps_relative(attack_df_2, timestamp, insert_timestamp)
            #print(last_timestamp)
            normal_df_temp = normal_df[(normal_df[timestamp] > old_last_timestamp) & 
            (normal_df[timestamp] < last_timestamp)]
        
            merged_df = self.merge(normal_df_temp, attack_df_2, timestamp, features)
        
            merged_df[features].to_csv(o_path, mode='a', index=False, header=not pd.io.common.file_exists(o_path)) 
        
        if test_train == 'test':
            np.save(self.labels + '/test_multi_label', merged_df['label'].to_numpy())
            np.save(self.labels + '/test_timestamp', merged_df[timestamp])
            test_y = merged_df['label'].to_numpy()
            np.save(self.labels + '/test_y', test_y)
            
    def shorten_attack_duration(self, df, timestamp_col, start_time):
        
        random_duration = pd.Timedelta(seconds=random.uniform(2, 10))
        if start_time == 0:
            start_time = df[timestamp_col].iloc[0]
            end_time = start_time + random_duration 
        if start_time != 0:
            df = df[df[timestamp_col] > start_time]
            start_time = df[timestamp_col].iloc[0]
            end_time = start_time + random_duration
            
        return end_time
        
    def build_new_dataset(self, feature_sel=True, num_features=100, timestamp='layers.frame.frame.time' ):

        if feature_sel:
            ranked_features = self.aggregate_feature_votes(self.ntrain_csvs_p)
            #np.save(os.path.join(self.files, 'train_test/all_features_ranked.npy'), ranked_features)        
            best_features = [feat for feat, count in sorted(ranked_features, key=lambda x: x[1], reverse=True)[:num_features]]
            best_features.append(timestamp)
            best_features_array = np.array(best_features)
            np.save(os.path.join(self.train_test, 'features.npy'), best_features_array)
 
        self.create_attack_sequences('train')
        self.create_attack_sequences('test')
    
        sequences_indexes = self.extract_sequences('train')
        self.extract_sequences_info('train', sequences_indexes)
        sequences_indexes = self.extract_sequences('test')
        self.extract_sequences_info('test', sequences_indexes)
        
    def extract_sequences(self, test_train):
        
        i_path = self.out_test_p
        o_path = self.labels
        if test_train == 'train':
            i_path = self.out_train_p
        
        df = pd.read_csv(i_path)

        df['layers.frame.frame.time'] = pd.to_datetime(df['layers.frame.frame.time'], format='mixed', errors='coerce')
        
        invalid_rows = df[df['layers.frame.frame.time'].isna()]
        if not invalid_rows.empty:
            print("Invalid timestamps found:")
            print(invalid_rows)

        sequences_indexes = []
        current_sequence = []
        
        for index, row in df.iterrows():
            sequence_value = row['sequence']
            if pd.isna(sequence_value):
                continue
            if not current_sequence or sequence_value == df.at[current_sequence[-1], 'sequence']:
                current_sequence.append(index)
            else:
                sequences_indexes.append(current_sequence)
                current_sequence = [index]
        
        if current_sequence:
            sequences_indexes.append(current_sequence)
        
        print(len(sequences_indexes))

        with open(os.path.join(o_path, 'sequence_indexes_' + test_train + '.pkl'), "wb") as file:
            pickle.dump(sequences_indexes, file)

        return sequences_indexes

    def extract_sequences_info(self, test_train, sequences_indexes):
        # Calculate statistics for each sequence
        i_path = self.out_test_p
        o_path = self.train_test
        if test_train == 'train':
            i_path = self.out_train_p
            
        df = pd.read_csv(i_path)
        df['layers.frame.frame.time'] = pd.to_datetime(df['layers.frame.frame.time'], format='mixed', errors='coerce')
        sequence_stats = []
        for seq_indexes in sequences_indexes:
            label = df.at[seq_indexes[0], 'label']
            timestamps = df.loc[seq_indexes, 'layers.frame.frame.time']
            duration = (timestamps.max() - timestamps.min()).total_seconds()
            length = len(seq_indexes)
            sequence_stats.append({'label': label, 'duration': duration, 'length': length})
        
        # Convert to DataFrame for aggregation
        sequence_stats_df = pd.DataFrame(sequence_stats)
        
        # Group by label and calculate statistics
        result = sequence_stats_df.groupby('label').agg({
            'duration': ['mean', 'max', 'min', 'std'],
            'length': ['mean', 'max', 'min', 'std'],
            'label': 'count'  # Count the number of sequences per label
        }).reset_index()
        
        # Flatten MultiIndex columns
        result.columns = ['label', 'avg_duration', 'max_duration', 'min_duration', 'st_dur',
                          'avg_length', 'max_length', 'min_length', 'num_sequences', 'std_len']
        
        print(result)

        result.to_csv(os.path.join(self.files, 'train_test/sequences_info_' + test_train + '.csv'))
            
    def preprocess_and_select_features(self, df: pd.DataFrame):
        # Encode categorical columns (except time)
        print("preprocessing")
        list_column_string = df.select_dtypes(exclude=[np.number]).columns
        for col in list_column_string:
            if col != 'layers.frame.frame.time':
                df[col] = pd.Categorical(df[col]).codes
    
        df = df.drop([col for col in df.columns if 'time' in col], axis=1, errors='ignore')
        df = df.drop([col for col in df.columns if 'mdns' in col], axis=1, errors='ignore')
    
        df.replace([np.inf, -np.inf], -1, inplace=True)
        df.fillna(-1, inplace=True)
        df = df.dropna(thresh=1, axis=1)
    
        df = df.loc[:, df.nunique() > 1]
    
        if df.select_dtypes(include=[np.number]).shape[1] == 0:
            return set()
    
        selector = VarianceThreshold(threshold=0)
        try:
            selector.fit(df.select_dtypes(include=[np.number]))
        except ValueError:
            return set()
    
        variances = df.var().values
        ranked_indices = np.argsort(variances)[::-1]
        selected_features = df.columns[ranked_indices]
    
        selected_set = set(selected_features)
        selected_set.add('layers.frame.frame.time')
    
        corr_matrix = df.corr(numeric_only=True)
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.99)]
        selected_set = selected_set - set(to_drop)
    
        return selected_set
    
    def aggregate_feature_votes(self, folder):
    
        feature_counter = Counter()
    
        for file in folder:
            print(f"Processing: {file}")
            try:
                df = pd.read_csv(file, nrows=1)
                columns = df.columns
                
                pattern = r"^(?!.*(dns|\\r|\\n|:|expert)).*$"
                filtered_columns = [col for col in columns if re.match(pattern, col)]
                
                df = pd.read_csv(file, usecols=filtered_columns)
                
                selected = self.preprocess_and_select_features(df)
                feature_counter.update(selected)
            
            except Exception as e:
                print(f"Failed to process")
                continue
    
        print("\nMost frequently selected features:")
        ranked_features = feature_counter.most_common()
        for feat, count in ranked_features:
            print(f"{feat}: {count} times")
        
        return ranked_features












