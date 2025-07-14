import pandas as pd
import os
import glob
import time
from multiprocessing import Pool

CONFIG = {
    "inbound": {
        "input_folder": "inbound",
        "temp_base_folder": "temp_output_inbound",
        "final_output_folder": "individual_proxy_inbound",
        "columns_to_extract": [
            "Timestamp", "ProxyId",
            "response1xxForwardedCounter",
            "response2xxForwardedCounter",
            "response3xxForwardedCounter",
            "response4xxForwardedCounter",
            "response5xxForwardedCounter",
            "response400ForwardedCounter",
            "response404ForwardedCounter",
            "response408ForwardedCounter",
            "response424ForwardedCounter",
            "response429ForwardedCounter"
        ],
        "dtype_map": {
            'ProxyId': 'category',
            'response1xxForwardedCounter': 'Int32',
            'response2xxForwardedCounter': 'Int32',
            'response3xxForwardedCounter': 'Int32',
            'response4xxForwardedCounter': 'Int32',
            'response5xxForwardedCounter': 'Int32',
            'response400ForwardedCounter': 'Int32',
            'response404ForwardedCounter': 'Int32',
            'response408ForwardedCounter': 'Int32',
            'response424ForwardedCounter': 'Int32',
            'response429ForwardedCounter': 'Int32'
        }
    },
    "outbound": {
        "input_folder": "outbound",
        "temp_base_folder": "temp_output_outbound",
        "final_output_folder": "individual_proxy_outbound",
        "columns_to_extract": [
            "Timestamp", "ProxyId",
            "response1xxReceivedCounter",
            "response2xxReceivedCounter",
            "response3xxReceivedCounter",
            "response4xxReceivedCounter",
            "response5xxReceivedCounter",
            "response400ReceivedCounter",
            "response404ReceivedCounter",
            "response408ReceivedCounter",
            "response424ReceivedCounter",
            "response429ReceivedCounter"
        ],
        "dtype_map": {
            'ProxyId': 'category',
            'response1xxReceivedCounter': 'Int32',
            'response2xxReceivedCounter': 'Int32',
            'response3xxReceivedCounter': 'Int32',
            'response4xxReceivedCounter': 'Int32',
            'response5xxReceivedCounter': 'Int32',
            'response400ReceivedCounter': 'Int32',
            'response404ReceivedCounter': 'Int32',
            'response408ReceivedCounter': 'Int32',
            'response424ReceivedCounter': 'Int32',
            'response429ReceivedCounter': 'Int32'
        }
    }
}

def process_one_file(args):
    file_path, temp_base_folder, columns_to_extract, dtype_map = args
    try:
        day_name = os.path.basename(file_path).replace(".csv", "")
        day_output_folder = os.path.join(temp_base_folder, day_name)
        os.makedirs(day_output_folder, exist_ok=True)

        df = pd.read_csv(
            file_path,
            usecols=columns_to_extract,
            dtype=dtype_map,
            parse_dates=["Timestamp"],
            date_format="%Y-%m-%d %H:%M:%S"
        )

        for proxy_id, group_df in df.groupby("ProxyId", observed=True):
            safe_proxy_id = str(proxy_id).replace("/", "_").replace("\\", "_")
            output_path = os.path.join(day_output_folder, f"{safe_proxy_id}.csv")
            group_df.to_csv(output_path, index=False)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    else:
        print(f"Done: {file_path}")

def merge_one_proxy(proxy_file_and_paths, final_output_folder):
    proxy_file, file_list = proxy_file_and_paths
    try:
        combined_df = pd.concat([pd.read_csv(f) for f in file_list])
        combined_df.to_csv(os.path.join(final_output_folder, proxy_file), index=False)
        print(f"Merged: {proxy_file}")
    except Exception as e:
        print(f"Error merging {proxy_file}: {e}")

def merge_all_proxy_files_parallel(temp_base_folder, final_output_folder, num_processes):
    day_folders = sorted(os.listdir(temp_base_folder))
    proxy_map = {}

    for day_folder in day_folders:
        full_path = os.path.join(temp_base_folder, day_folder)
        for file in os.listdir(full_path):
            if file not in proxy_map:
                proxy_map[file] = []
            proxy_map[file].append(os.path.join(full_path, file))

    proxy_items = list(proxy_map.items())

    with Pool(processes=num_processes) as pool:
        pool.starmap(merge_one_proxy, [(item, final_output_folder) for item in proxy_items])

def run_preprocessing(direction, num_processes=8):
    if direction not in CONFIG:
        return f"Error: Unknown direction '{direction}'"

    cfg = CONFIG[direction]
    os.makedirs(cfg["temp_base_folder"], exist_ok=True)
    os.makedirs(cfg["final_output_folder"], exist_ok=True)

    all_files = sorted(glob.glob(os.path.join(cfg["input_folder"], "*.csv")))
    if not all_files:
        return f"Error: No files found in {cfg['input_folder']}"

    start_time = time.time()

    args_list = [
        (file_path, cfg["temp_base_folder"], cfg["columns_to_extract"], cfg["dtype_map"])
        for file_path in all_files
    ]

    with Pool(processes=num_processes) as pool:
        pool.map(process_one_file, args_list)

    merge_all_proxy_files_parallel(cfg["temp_base_folder"], cfg["final_output_folder"], num_processes)

    end_time = time.time()
    return f"Processed {len(all_files)} files in {end_time - start_time:.2f} seconds. Output: {cfg['final_output_folder']}"

