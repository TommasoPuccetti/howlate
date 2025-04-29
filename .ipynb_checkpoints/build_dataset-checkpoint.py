import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent / "src/utils"))
import loader as ld
sys.path.append(str(Path(__file__).resolve().parent / "src/dataset_builder"))
import dataset_builder as db

# Argument parsing
parser = argparse.ArgumentParser(description="Build a dataset for training and evaluation.")
parser.add_argument("dataset", type=str, help="Name of the dataset (e.g., dos_mqtt_iot)")

args = parser.parse_args()


# Initialize path manager
print(f"Initializing path manager for dataset: {args.dataset}")
pm = ld.PathManager(args.dataset, 'None')

# Initialize dataset builder
print("Building dataset...")
dbr = db.DatasetBuilder(pm)
dbr.build_new_dataset()

print("Dataset built successfully.")