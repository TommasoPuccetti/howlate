"""
This module evaluates an Intrusion Detection model by computing standard classification metrics and the time an attack remains undetected. The evaluation includes the following key metrics:

- **False Positive Rate (FPR):** The proportion of normal instances misclassified as attacks.
- **Attack Latency (Δl):** The time taken to detect each attack sequence.
- **Sequence Detection Rate (SDR):** The number of detected attack sequences over the total attack sequences.

### Evaluation Approach
To measure the average latency (ΔL) and SDR at different FPR levels, the evaluation keeps track of:

1. **Initial Data Point of Each Attack Sequence:** Marks the start of each attack.
2. **Positions of Data Points Labeled as Anomalous:** Tracks where the model detects anomalies.
3. **First Correctly Classified Anomalous Data Point in Each Sequence:** Determines when an attack is first detected.

These metrics are only meaningful if the dataset consists of sequences containing normal and anomalous operations.
"""

import os
import argparse
import numpy as np
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import warnings
import shutil
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / "./src/utils"))
from config import *
import loader as ld
import plotter as pl
sys.path.append(str(Path(__file__).resolve().parent / "./src/evaluator"))
from evaluator_latency import LatencyEvaluator
from evaluator_sota import SotaEvaluator



warnings.filterwarnings('ignore')

# ----------------------
# Argument Parser
# ----------------------
parser = argparse.ArgumentParser(description="Evaluate IDS model performance")
parser.add_argument("dataset", type=str, help="Name of the dataset (e.g., dos_mqtt_iot)")
parser.add_argument("model", type=str, help="Name of the model used (e.g., extra, xgboost, lstm)")
parser.add_argument("--verbose", action="store_true", help="Print detailed logs")

args = parser.parse_args()
#TODO implement a verbose output 
# ----------------------
# Load Dataset and Model
# ----------------------
print(f"Loading dataset '{args.dataset}' with model '{args.model}'...")
pm = ld.PathManager(args.dataset, args.model)
dl = ld.DataLoader(pm)

# ----------------------
# Clean result folder
# ----------------------

result_dir = pm.results_p
if os.path.exists(result_dir):
    print(f"Cleaning results directory: {result_dir}")
    shutil.rmtree(result_dir)
os.makedirs(result_dir, exist_ok=True)

# ----------------------
# Initialize Evaluators
# ----------------------
latency = LatencyEvaluator(pm)
sota = SotaEvaluator(pm)

# ----------------------
# SOTA Evaluation
# ----------------------
print("Evaluating SOTA metrics...")
sota_results_fprs = sota.evaluate(dl.test_y, dl.test_multi, dl.preds_proba)
sota.evaluate_bin_preds(dl.test_y, dl.preds)

# ----------------------
# Plot ROC & PR curves
# ----------------------
print("Plotting curves...")
sota.plot_curves(dl.test_y, dl.preds_proba)

# ----------------------
# Latency Evaluation
# ----------------------
print("Evaluating latency metrics...")
avg_results, tradeoff_summary = latency.evaluate(
    dl.test_y, dl.test_multi, dl.test_timestamp, dl.test_seq, dl.preds_proba
)

# ----------------------
# Plotter
# ----------------------
print("Generating latency plots...")
plot = pl.Plotter(pm, args.model)
plot.plot()

print("Evaluation complete.")




    



