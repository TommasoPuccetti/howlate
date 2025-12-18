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



## References

- [CIC-IoT-2023 Dataset - Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/iot2023.html)
- [Wireshark editcap tool](https://www.wireshark.org/docs/man-pages/editcap.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
