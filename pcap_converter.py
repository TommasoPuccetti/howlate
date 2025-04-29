#!/usr/bin/env python

import argparse, os, json, sys
from pathlib import Path
import subprocess as sub
import pandas as pd
import ijson

def convert_pcap_to_csv(pcap_file, output_dir, chunk_size=1000):
    """Efficiently convert a PCAP file to CSV using tshark and streaming JSON parsing."""
    try:
        proc = sub.Popen(['tshark', '-T', 'json', '-r', str(pcap_file)], stdout=sub.PIPE)
        parser = ijson.items(proc.stdout, 'item._source')
        
        rows = []
        output_file = output_dir / f'{pcap_file.stem}.csv'

        for packet in parser:
            rows.append(pd.json_normalize(packet))
            if len(rows) >= chunk_size:
                df = pd.concat(rows, ignore_index=True)
                df.to_csv(output_file, mode='a', index=False, header=not output_file.exists())
                rows.clear()

        if rows:
            df = pd.concat(rows, ignore_index=True)
            df.to_csv(output_file, mode='a', index=False, header=not output_file.exists())

        print(f"Processed: {pcap_file.name} -> {output_file}")
    
    except Exception as e:
        print(f"Error processing {pcap_file.name}: {e}")

def resolve_dataset_paths(dataset_name):
    """Returns list of valid subfolders (normal/attacks) under the dataset path."""
    base = Path("./data") / dataset_name / "dataset" / "raw" / "pcaps"
    normal = base / "normal/split"
    attacks = base / "attacks/split"
    return [p for p in [normal, attacks] if p.exists()]

def get_output_path(input_folder: Path, dataset_name: str):
    """Determine output path based on whether 'normal' or 'attacks' is in the path."""
    if 'attacks' in input_folder.parts:
        return Path(f"./data/{dataset_name}/dataset/converted/attacks")
    else:
        return Path(f"./data/{dataset_name}/dataset/converted/normal")

def process_folder(input_folder: Path, dataset_name: str):
    """Convert all PCAPs in the folder and store them in the correct converted subfolder."""
    output_dir = get_output_path(input_folder, dataset_name)
    output_dir.mkdir(parents=True, exist_ok=True)


    for pcap_file in input_folder.glob("*.pcap*"):
        if pcap_file.is_file():
            convert_pcap_to_csv(pcap_file, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PCAP[ng] files to CSV using tshark.")
    parser.add_argument("--input-folder", type=str, help="Path to folder containing PCAPs.")
    parser.add_argument("--output-folder", type=str, help="Path to output folder (used with --input-folder).")
    parser.add_argument("--dataset-name", type=str, help="Dataset name (e.g., cic_iot_23).")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Number of packets to process per chunk (default=1000).")
    args = parser.parse_args()

    if args.input_folder:
        input_folder = Path(args.input_folder).resolve()
        if not input_folder.exists() or not input_folder.is_dir():
            print(f"Input folder does not exist: {input_folder}")
            sys.exit(1)
        
        if not args.output_folder:
            print("You must specify --output-folder when using --input-folder.")
            sys.exit(1)
        
        output_folder = Path(args.output_folder).resolve()
        output_folder.mkdir(parents=True, exist_ok=True)

        for pcap_file in input_folder.glob("*.pcap*"):
            if pcap_file.is_file():
                convert_pcap_to_csv(pcap_file, output_folder)


    elif args.dataset_name:
        dataset_name = args.dataset_name
        folders = resolve_dataset_paths(dataset_name)
        if not folders:
            print(f"No valid raw PCAP folders found for dataset '{dataset_name}'.")
            sys.exit(1)
        for folder in folders:
            print(f"\n Processing folder: {folder}")
            process_folder(folder, dataset_name)

    else:
        print("Error: Please provide either --input-folder or --dataset-name.")
        parser.print_help()
