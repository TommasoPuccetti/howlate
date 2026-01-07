# HOWLATE Framework

HOWLATE is a framework designed to evaluate the timeliness of intrusion detection systems alongside traditional performance metrics. Specifically, it measures attack latency, defined as the time from the start of an attack to the time it is detected. More details can be found in the related conference paper [R] 

The tool can:
1) Evaluate a model's performance based on its predictions.
2) Create NID datasets from existing network traces (.pcaps) 

In the following, we resume the different steps from the creation of a dataset to the evaluation of a trained model. The steps can be operated in isolation just by using the appropriate scripts and files. 

---

## Installation 

To install the script the only steps are to create and activate a conda environment using the howlate_env.yml file. The folowing commands will install the required dependecy to run:

1) conda env create -f howlate_env.yml
2) conda activate latency

To replicate the results of the paper you can use the ./data folder from https://drive.google.com/drive/folders/1kUnCcT_ZLSUNPyLF8bN3ycsmG4K066GX?usp=sharing

## 1) Process Raw Data

### 1a) Split PCAPs (Optional)

Use `editcap` from Tshark to split large PCAPs. Each split file from an **attack PCAP** will generate **two sequences** in the final dataset.

> The number of attack PCAPs determines the number of attack sequences.  
> Example: 3 split files = 6 attack sequences.

---

### 1b) Convert PCAPs to CSV

Use `convert.py` to transform PCAPs into CSV files.

#### Usage

```bash
# a) Convert all PCAPs in a specific folder
python convert.py \
  --input-folder ./data/cic_iot_23/dataset/raw/pcaps/normal \
  --output-folder ./data/cic_iot_23/dataset/raw/converted/normal

# b) Convert using dataset name (handles both normal and attack folders)
python convert.py --dataset-name cic_iot_23
```

- Output:
  - `./data/<dataset_name>/dataset/raw/converted/normal/`
  - `./data/<dataset_name>/dataset/raw/converted/attacks/`

---

## 2) Build the Dataset

### 2a) Place Input Data

Organize the converted CSVs into the following folder structure:

```
./data/<dataset_name>/dataset/raw/converted/
├── normal/
│   ├── train/   # One CSV file with normal training traffic
│   └── test/    # One CSV file with normal testing traffic
└── attacks/
    ├── train/   # Multiple CSVs; each file = 2 sequences
    └── test/    # Multiple CSVs; each file = 2 sequences
```

> 🛠 If folders are missing, create them manually.

---

### 2b) Dataset Builder

Run `build_dataset.py` to merge and label data into a time-series classification dataset.

#### Usage

```bash
python build_dataset.py <dataset_name>

# Example:
python build_dataset.py cic_iot_23
```

#### Output: `./data/<dataset_name>/dataset/train_test/`

- `train_set.csv` — Time-series training set
- `test_set.csv` — Time-series test set
- `sequence_indexes_train.pkl` — List of packet indices for each attack sequence in training
- `sequence_indexes_test.pkl` — List of packet indices for each attack sequence in test
- `sequence_info_train.csv` — Metadata for each sequence in training
- `sequence_info_test.csv` — Metadata for each sequence in test

---

## 3) Train a Model (Optional)

Model training is **not included** in the framework, but a sample notebook is provided:

- `train_xgb_cic_iot_23.ipynb`  
  An example using **XGBoost** to train a binary classifier on the generated dataset from CIC-IoT-2023.

---

## 4) Evaluation

Use `evaluate.py` to evaluate the performance of your trained model.

#### Usage

```bash
python evaluate.py <dataset_name> <model_name>

# Example:
python evaluate.py cic_iot_23 xgb
```

#### Required Files (in `dataset/train_test/` folder):

- `test_y.npy` — Binary ground-truth labels (0 = normal, 1 = attack)
- `test_multi_label.npy` — Multi-class attack labels (e.g., `icmp_dos`, `scan`)
- `test_timestamp.npy` — Capture timestamps for each packet
- `test_sequences.pkl` — List of packet indices for each attack sequence

#### Required Files (in `models/model_name/preds/` folder):
- `preds.npy` — Final binary predictions from the model
- `preds_proba.npy` — Prediction probabilities from the model
---

<img width="1352" height="1180" alt="image" src="https://github.com/user-attachments/assets/aafb92ed-e08d-4101-9f78-a2ee68ed271c" />

## 5) Practical Example

We demonstrate how HOWLATE operates using raw network trace packets from the CIC IoT 2023 dataset (hereafter referred to as cic_iot_23) to construct a new dataset and evaluate a detector. The cic_iot_23 dataset executes a network topology composed of 105 different IoT devices, which are subjected to 33 different attacks. The dataset adopts a flow-based approach, where each monitored communication session is represented using summarizing features such as flow duration, protocol type, and other aggregated network statistics. As explained in Section II.A, this format is not suitable for computing attack latency. To address this limitation, we leverage the available network traffic in pcap format to construct a dataset that enables the measurement of attack latency. Specifically, we select three representative attacks from the original dataset and compose a new dataset consisting of both normal and attack packet sequences. 
HOWLATE assumes the default folder structure shown in Figure above. Input files must be placed in the correct directories 
for the tool to run successfully. The following section outlines the required files, their locations, and how HOWLATE utilizes them.

### Process Raw Data

The pcap_converter.py script takes as input a folder containing pcap files and converts them into csv files. The script reports all the features from the captured network packets in the pcap files using Tshark [27]. The conversion may be resource-consuming, depending on the pcap files. The tool allows performing the conversion in batches to optimize the process. The chunk-size argument specifies the number of packets in each batch. To further optimize the process, the pcaps can be split using the editcaps tool from Tshark [27].

### Dataset Builder

The build_dataset.py script composes a csv dataset using the converted pcap files. The script extracts two sequences of random length from each of the provided attack csv files and merges them with the normal traffic.
The minimum and maximum sequence lengths for both train and test can be specified using the config file in the ./src/utils/ folder, along with the maximum and minimum pause between sequences. The merging is performed separately for test and train, and outputs separate test and train sets as csv files. 
As reported in the central box in the Figure, the script uses normal traffic files in the ./data/raw/converted/normal/test and ./data/raw/converted/normal/train. In Fig. 4, we report files as examples in the specified paths. When launching the script, two sequences of packets are extracted from the SYN-DDoS-AD_1.csv and merged with normal traffic in the BenignTraffic_1.csv. The output is the train_set.csv file in the ./data/cic_iot_23/dataset/train_test folder. An analogous process is performed on the test sets file, BenignTraffic_2.csv, and SYN-DDoS-AD_1.csv to compose the test_set.csv file. 
Additionally, the script produces the following files:
test_y.npy: the binary labels of the test set indicating if a packet belongs to an attack sequence or normal traffic.
test_multi_label.npy:  the multi-label of the test set indicating the type of attacks, e.g., icmp_dos. 
test_timestamp.npy: the timestamps of the test set specifying for each packet its time instant ti.
test_sequences.pkl: for each attack sequence, it reports a list of indexes to keep track of all the attack packets in the sequence.  
sequence_info_train.csv and sequence_info_test.csv: these files contain metadata about the sequences created. In Table I, we exemplify the content of these files using the cic_iot_23 datasets created with the tool.

### Model Evaluation

To demonstrate how HOWLATE evaluation works, we trained a NID solution based on XGBoost, an algorithm designed for supervised learning tasks that combines the prediction of a regression trees ensemble. 
The evaluate.py script evaluates the NID performance based on state-of-the-art metrics and latency-related metrics. 
The script takes as input the test_y.npy, test_multi_label.npy, test_timestamp.npy, and test_sequences.pkl files generated during the dataset composition. Additionally, the script requires the preds_proba.npy file containing the prediction probabilities produced by the detector model during testing.
 For the script to function correctly, the required files must be placed in the appropriate folders as specified in figure. Briefly, all labels, timestamps, and sequence indices must be placed in the ./data/cic_iot_23/labels folder, and the files reporting model predictions must be placed in the ./data/cic_iot_23/models/xgb/preds folder. The evaluate.py script takes these names as input to refer to the related files and to perform the evaluation. The results files will be placed in ./data/cic_iot_23/xgb/results/ and are the following: 
i) precision_recall_curve.pdf and roc_curve.pdf are the precision-recall and ROC curves for the model. We report the resulting curves in Fig. 5.  
ii) fpr_value_sota.csv: these files report the P, R, F1, TP, TN, FP, and FN obtained with the target FPR value specified in the file name, e.g., 0.01_sota.csv contains the above metrics measured with FPR=0.01. A sample of this table is reported in Table II. 
iii) fpr_value_latency.xlsx: reports latency and SDR. This file contains multiple parts (sheets):
all_sequences_results reports information on the sequences, in particular the start and end indexes of each attack, the attack duration in seconds, the attack latency, the index at which the attack is detected (if any), the attack sequences length in number of packets, the attack type, if the sequence is detected or not, and the R. An example of this sheet is reported in Table III.  
detected_sequences_results details the same information but reports only detected sequences.
avg_results_for_attack_type reports the average and standard deviation for some of the metrics reported in the all_sequences_results sheet, aggregated by attack type. We detail this in Table IV. 
detection_rate_for_attack_type reports, for each attack type, the number of sequences, the number of detected sequences, and the SDR. We provide a sample in Table V. 
overall_result sheet reports the same metrics of the detection_rate_for_attack_type but on all attacks, along with the average, standard deviation, minimum, and maximum of the time to detect, and the index of detection.
detection_rate_overall reports the number of detected sequences, the total number of sequences, and the percentage of detected sequences. 
iv) final_results.pdf: plots the trade-off between FPR, attack latency, and SDR. As an example, we report in Fig. 6  the plot related to XGBoost on the cic_iot_23 dataset. 

## References

- [CIC-IoT-2023 Dataset - Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/iot2023.html)
- [Wireshark editcap tool](https://www.wireshark.org/docs/man-pages/editcap.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
