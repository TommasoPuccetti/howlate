"""
This module define the PathManager class that handles the input and output path fof operating the files needed by the framework. Additionally define the DataLoader class that uses the paths of a defined PathManager to load and output the files to perform detector evaluation and dataset creation 

"""

import numpy as np
import pandas as pd 
import os
from config import *
import results_handler as rh 

class PathManager:
    
    # === PathManager ===
    
    """
    Initialize the paths to files needed for evaluation.
     
     Attributes:
         - **root:** data root folder.
         - **dataset:** name of the dataset.
         - **models:** name of the model.
         - **dataset_root:** points to dataset folder e.g. (./data/dos_mqtt_iot).
         - **files:** points to the "dataset" folder of dataset_root (contains the dataset and label folder)
         - **models:** points to the folder where folders dedicated to models are stored.
         - **target_model:** points to a specific dataset folder (contains model and prediction folder)
         - **test_y_p:** points to test binary label.
         - **test_multi_p:** points to test multi-class label.
         - **timestamp_test_p:** points to test timestamp.
         - **test_seq_p:** points to test sequences indices.
         - **preds_proba_p:** points to model prediction probabilities.
         - **preds_p:** points to model binary predictions.
    """
    def __init__(self, dataset, model, root=ROOT, verbose=False):
        self.root = root
        self.dataset = dataset
        self.model = model
        self.dataset_root = os.path.join(self.root, self.dataset)
        self.files = os.path.join(self.dataset_root, "dataset")
        self.models = os.path.join(self.dataset_root, "models")
        self.target_model = os.path.join(self.models, self.model)
        rh.create_folder_if_not_exist(self.files)
        rh.create_folder_if_not_exist(self.target_model)
        if verbose:
            print_paths()

    @property
    def labels(self):
        path = os.path.join(self.files, "labels")
        rh.create_folder_if_not_exist(path)
        return path
    @property
    def test_y_p(self):
        return os.path.join(self.labels, "test_y.npy")
    @property
    def test_multi_p(self):
        return os.path.join(self.labels, "test_multi_label.npy")
    @property
    def test_timestamp_p(self):
        return os.path.join(self.labels, "test_timestamp.npy")
    @property
    def test_seq_p(self):
        return os.path.join(self.labels, "test_sequences.pkl")
    @property
    def preds(self):
        path = os.path.join(self.target_model, "preds")
        rh.create_folder_if_not_exist(path)
        return path
    @property
    def preds_proba_p(self):
        return os.path.join(self.preds, "preds_proba.npy")
    @property
    def preds_p(self):
        return os.path.join(self.preds, "preds.npy")
    @property
    def results_p(self):
        path = os.path.join(self.target_model, "results")
        rh.create_folder_if_not_exist(path)
        return path
    @property
    def raw_converted_n(self):
        path = os.path.join(self.files, "raw/converted/normal")
        rh.create_folder_if_not_exist(path)
        return path
    @property
    def raw_converted_a(self):
        path = os.path.join(self.files, "raw/converted/attacks")
        rh.create_folder_if_not_exist(path)
        return path
    @property
    def ntest_csv_p(self):
        return os.path.join(self.raw_converted_n, "test_merged.csv")
    @property
    def ntrain_csv_p(self):
        return os.path.join(self.raw_converted_n, "train_merged.csv")
    @property
    def ntest_csvs_p(self):
        normal_root_test = os.path.join(self.files, "raw/converted/normal/test")
        rh.create_folder_if_not_exist(normal_root_test)
        files = os.listdir(normal_root_test)
        normal_files_test = [file for file in files if file.endswith('.csv') and file]
        normal_files_test_p = [os.path.join(normal_root_test, file) for file in normal_files_test]
        return normal_files_test_p
    @property
    def ntrain_csvs_p(self):
        attacks_root_train = os.path.join(self.files, "raw/converted/normal/train")
        rh.create_folder_if_not_exist(attacks_root_train)
        files = os.listdir(attacks_root_train)
        attack_files_train = [file for file in files if file.endswith('.csv') and file]
        attack_files_train_p = [os.path.join(attacks_root_train, file) for file in attack_files_train]
        return attack_files_train_p
    @property
    def atest_csvs_p(self):
        attacks_root_test = os.path.join(self.files, "raw/converted/attacks/test")
        rh.create_folder_if_not_exist(attacks_root_test)
        files = os.listdir(attacks_root_test)
        attack_files_test = [file for file in files if file.endswith('.csv') and file]
        attack_files_test_p = [os.path.join(attacks_root_test, file) for file in attack_files_test]
        return attack_files_test_p
    @property
    def atrain_csvs_p(self):
        attacks_root_train = os.path.join(self.files, "raw/converted/attacks/train")
        rh.create_folder_if_not_exist(attacks_root_train)
        files = os.listdir(attacks_root_train)
        attack_files_train = [file for file in files if file.endswith('.csv') and file]
        attack_files_train_p = [os.path.join(attacks_root_train, file) for file in attack_files_train]
        return attack_files_train_p
    @property
    def train_test(self):
        path = os.path.join(self.files, "train_test")
        rh.create_folder_if_not_exist(path)
        return path
    @property
    def out_train_p(self):
        return os.path.join(self.train_test, "train_set.csv")
    @property
    def out_test_p(self):
        return os.path.join(self.train_test, "test_set.csv")
        
    def print_paths():
        print("Dataset:  {}, \nModel:  {}".format(preds, self.model))
        print("Dataset path:  {}, \nModel path:  {}".format(self.files, self.target_model))
        print("test_y path:  {}, \ntest_seq path:  {}, \ntest_multi path {}".format(test_y_p, test_seq_p, test_multi_p))
        print("preds_proba path:  {}, \npreds path:  {}".format(preds_proba_p, preds))
        print("Converted normal test path:  {}, \ntrain path:  {}".format(ntrain_csv_p, ntest_csv_p))
        print("Converted attacks test path:  {}, \ntrain path:  {}".format(atrain_csv_p, atest_csv_p))
        print(PRINT_SEPARATOR)


class DataLoader:
    
    # === DataLoader
    
    """
    Load files needed for evaluation.
    
     Attributes:
     
         - **path_manager:** an instance of the PathManager class that points to data to load.
         - **test_y:** .npy binary label.
         - **test_multi:** .npy multi-class label.
         - **timestamp_test:** .npy test timestamp.
         - **test_seq:** .npy test sequences indices.
         - **preds_proba:** .npy prediction probabilities of the model.
         - **preds:** .npy binary predictions of the model.
    """
    def __init__(self, paths: PathManager):
        self.paths = paths
    
    @property
    def test_y(self):
        return np.load(self.paths.test_y_p, allow_pickle=True)
    @property
    def test_multi(self):
        return np.load(self.paths.test_multi_p, allow_pickle=True)
    @property
    def test_timestamp(self):
        return np.load(self.paths.test_timestamp_p, allow_pickle=True)
    @property
    def test_seq(self):
        return np.load(self.paths.test_seq_p, allow_pickle=True)
    @property
    def preds_proba(self):
        return np.load(self.paths.preds_proba_p, allow_pickle=True)
    @property
    def preds(self):
        return np.load(self.paths.preds_p, allow_pickle=True)