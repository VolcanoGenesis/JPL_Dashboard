import os
import glob
import multiprocessing
import time
from datetime import datetime

from ag import detect_anomalies, filter_anomalies_df

def get_user_choices():
    while True:
        dir_choice = input("Enter input direction (inbound/outbound): ").strip().lower()
        if dir_choice in ("inbound", "outbound"):
            break
        print("Invalid input. Please enter 'inbound' or 'outbound'.")
    if dir_choice == "inbound":
        input_dir = "individual_proxy_inbound"
        output_dir_base = "anomaly_output_inbound"
        counter_map = {
            "2xx": "response2xxForwardedCounter",
            "4xx": "response4xxForwardedCounter",
            "5xx": "response5xxForwardedCounter"
        }
    else:
        input_dir = "individual_proxy_outbound"
        output_dir_base = "anomaly_output_outbound"
        counter_map = {
            "2xx": "response2xxReceivedCounter",
            "4xx": "response4xxReceivedCounter",
            "5xx": "response5xxReceivedCounter"
        }
    while True:
        counter_choice = input("Enter counter (2xx/4xx/5xx): ").strip().lower()
        if counter_choice in counter_map:
            break
        print("Invalid input. Please enter '2xx', '4xx' or '5xx'.")
    column_name = counter_map[counter_choice]
    output_dir = f"{output_dir_base}_{counter_choice}"
    # Prompt for date range
    start_date_str = input("Enter start date (YYYY-MM-DD) or leave blank for earliest: ").strip()
    end_date_str = input("Enter end date (YYYY-MM-DD) or leave blank for latest: ").strip()
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d") if start_date_str else None
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d") if end_date_str else None
    return input_dir, output_dir, column_name, start_date, end_date

def process_file(args):
    file_path, column_name, output_dir, plot_dir, start_date, end_date = args
    try:
        print(f"Processing: {file_path}")
        anomaly_df = detect_anomalies(file_path, column_name, plot_dir, start_date, end_date)

        base_name = os.path.basename(file_path).replace(".csv", "_anomalies_filtered.csv")
        final_output = os.path.join(output_dir, base_name)

        filter_anomalies_df(anomaly_df, final_output, column_name)

        print(f"Done: {file_path}")
        return final_output
    except Exception as e:
        print(f"Failed processing {file_path}: {e}")
        return None

INPUT_DIR, OUTPUT_DIR, COLUMN_NAME, START_DATE, END_DATE = get_user_choices()
os.makedirs(OUTPUT_DIR, exist_ok=True)
all_files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
print(f"Found {len(all_files)} files in {INPUT_DIR}.")

plot_dir = OUTPUT_DIR.replace("anomaly_output", "anomaly_plots")
os.makedirs(plot_dir, exist_ok=True)

cpu_count = multiprocessing.cpu_count()
try:
    user_input = input(f"Enter number of processes to use (1-{cpu_count}, default={cpu_count}): ")
    num_processes = int(user_input) if user_input.strip() else cpu_count
    if num_processes < 1 or num_processes > cpu_count:
        print(f"Invalid input. Using default: {cpu_count}")
        num_processes = cpu_count
except Exception:
    print(f"Invalid input. Using default: {cpu_count}")
    num_processes = cpu_count

print(f"Using {num_processes} CPU cores for multiprocessing.")

start_time = time.time()

# Pass date range to process_file
args_list = [(file_path, COLUMN_NAME, OUTPUT_DIR, plot_dir, START_DATE, END_DATE) for file_path in all_files]

with multiprocessing.Pool(num_processes) as pool:
    results = pool.map(process_file, args_list)

end_time = time.time()
elapsed = end_time - start_time

successful = [r for r in results if r]
print(f"\nCompleted processing {len(successful)} files. Output stored in '{OUTPUT_DIR}'.")
print(f"Total execution time: {elapsed:.2f} seconds.")