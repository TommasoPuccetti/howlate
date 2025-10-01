#!/usr/bin/env python

import argparse, os, json, sys
from pathlib import Path
import subprocess as sub
import pandas as pd
import json
import tempfile


def convert_pcap_to_csv(pcap_file, output_dir):
    """Convert a PCAP file to CSV using tshark -T json (safe), skipping malformed packets."""
    try:
        output_file = output_dir / f'{pcap_file.stem}.csv'
        rows = []

        mqtt_ports = [
    1883, 32845, 33121, 33179, 33297, 33801, 33941, 34082, 34115, 34121, 34337, 34367, 34599, 34927, 35009, 35021, 35041,
    35247, 35331, 35399, 35457, 35571, 35617, 35634, 35671, 35673, 35691, 35827, 35839, 35851, 35859, 35908, 35966, 36095,
    36151, 36209, 36215, 36305, 36349, 36353, 36489, 36503, 36511, 36531, 36563, 36629, 36679, 36869, 37081, 37213, 37217,
    37371, 37375, 37605, 37673, 37985, 38065, 38281, 38409, 38593, 38599, 38681, 39013, 39017, 39187, 39389, 39393, 39431,
    39435, 39627, 39689, 39791, 39973, 39993, 40161, 40262, 40414, 40421, 40489, 40495, 40499, 40541, 40543, 40547, 40569,
    40585, 40591, 40617, 40625, 40637, 40649, 40667, 40916, 41043, 41081, 41089, 41133, 41407, 41501, 41735, 41795, 41895,
    41977, 41985, 41999, 42012, 42135, 42273, 42299, 42336, 42351, 42367, 42697, 42827, 42833, 42835, 43201, 43274, 43275,
    43301, 43359, 43589, 43912, 43999, 44227, 44249, 44407, 44477, 44911, 44953, 45157, 45185, 45515, 45591, 45635, 45649,
    45665, 45945, 45972, 46231, 46300, 46357, 46359, 46669, 46745, 46753, 46777, 46851, 47139, 47147, 47217, 47221, 47277,
    47306, 47527, 47665, 47739, 47789, 47889, 47915, 48137, 48173, 48247, 48267, 48471, 48533, 48553, 48794, 49015, 49115,
    49173, 49200, 49274, 49405, 49439, 49631, 49633, 49749, 49787, 50094, 50166, 50331, 50345, 50519, 50521, 50837, 50891,
    50913, 50937, 51175, 51211, 51225, 51467, 51583, 51903, 52043, 52159, 52387, 52445, 52503, 52528, 52619, 52679, 52855,
    52963, 52983, 52989, 53021, 53105, 53229, 53265, 53319, 53497, 53537, 53627, 53631, 53639, 53691, 53715, 53717, 53719,
    53795, 53808, 53853, 53897, 54038, 54065, 54091, 54179, 54495, 54591, 54713, 55177, 55812, 55856, 55887, 55965, 56599,
    56629, 56775, 56805, 56873, 56893, 56903, 57019, 57072, 57079, 57203, 57231, 57251, 57276, 57389, 57429, 57485, 57609,
    57623, 57729, 57767, 57861, 58227, 58363, 58513, 58723, 58773, 58791, 58815, 58915, 58972, 58973, 58985, 58989, 59147,
    59281, 59567, 59647, 59805, 59823, 59829, 59883, 59901, 59955, 59973, 60103, 60193, 60313, 60335, 60345, 60641, 60672,
    60707, 60923, 60962
]

        tshark_cmd = ['tshark', '-r', str(pcap_file), '-T', 'json']
        for port in mqtt_ports:
            tshark_cmd += ['-d', f'tcp.port=={port},mqtt']

        print("📦 Running command:", ' '.join(tshark_cmd))

        with tempfile.NamedTemporaryFile(delete=False, mode='w+', encoding='utf-8') as tmpfile:
            proc = sub.Popen(tshark_cmd, stdout=tmpfile, stderr=sub.PIPE, text=True)
            stderr = proc.stderr.read()
            proc.wait()

            if proc.returncode != 0:
                raise RuntimeError(f"tshark failed with code {proc.returncode}:\n{stderr}")

            tmpfile_path = tmpfile.name

        with open(tmpfile_path, 'r', encoding='utf-8') as f:
            try:
                packets = json.load(f)
            except json.JSONDecodeError as e:
                print(f"❌ JSON corrupted. Tshark likely wrote broken JSON: {e}")
                return

            for obj in packets:
                try:
                    packet = obj["_source"]
                    flat = pd.json_normalize(packet, sep='_')
                    rows.append(flat)
                except Exception as e:
                    print(f"⚠️ Skipping malformed packet: {e}")
                    continue

        if rows:
            df = pd.concat(rows, ignore_index=True)
            df.to_csv(output_file, index=False)

        print(f"✅ Processed: {pcap_file.name} → {output_file}")

    except Exception as e:
        print(f"❌ Error processing {pcap_file.name}: {e}")

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
