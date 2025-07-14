import pandas as pd
import os
import glob
import time
from multiprocessing import Pool, cpu_count

input_folder = "inbound"
temp_base_folder = "temp_output_inbound"
final_output_folder = "individual_proxy_inbound"

os.makedirs(temp_base_folder, exist_ok=True)
os.makedirs(final_output_folder, exist_ok=True)

columns_to_extract = [
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
]

dtype_map = {
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


def process_one_file(file_path):
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


def merge_one_proxy(proxy_file_and_paths):
    proxy_file, file_list = proxy_file_and_paths
    try:
        combined_df = pd.concat([pd.read_csv(f) for f in file_list])
        combined_df.to_csv(os.path.join(final_output_folder, proxy_file), index=False)
        print(f"Merged: {proxy_file}")
    except Exception as e:
        print(f"Error merging {proxy_file}: {e}")


def merge_all_proxy_files_parallel():
    day_folders = sorted(os.listdir(temp_base_folder))
    proxy_map = {}

    for day_folder in day_folders:
        full_path = os.path.join(temp_base_folder, day_folder)
        for file in os.listdir(full_path):
            if file not in proxy_map:
                proxy_map[file] = []
            proxy_map[file].append(os.path.join(full_path, file))

    proxy_items = list(proxy_map.items())

    with Pool(processes=80) as pool:
        pool.map(merge_one_proxy, proxy_items)


if __name__ == "__main__":
    start_time = time.time()

    all_files = sorted(glob.glob(os.path.join(input_folder, "*.csv")))
    print(f"Found {len(all_files)} files.")

    with Pool(processes=80) as pool:
        pool.map(process_one_file, all_files)

    print("\nMerging all proxy outputs in parallel...")
    merge_all_proxy_files_parallel()

    end_time = time.time()
    print(f"\nDone. Total time: {end_time - start_time:.2f} seconds")
