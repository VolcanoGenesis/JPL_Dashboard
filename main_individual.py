from anomalyisowithmonthend import detect_anomalies
from filteringusingrollingmean import filter_anomalies
import os
import time
from collections import defaultdict

start_time = time.time()

def get_all_proxies(data_folder):
    proxies = []
    for fname in os.listdir(data_folder):
        if fname.endswith('.csv'):
            proxies.append(fname[:-4])
    return proxies

def build_proxy_hierarchy(all_proxies):
    hierarchy = defaultdict(lambda: defaultdict(list))
    for proxy in all_proxies:
        parts = proxy.split('_')
        city = parts[-1]
        nf_type = parts[0]
        if "IngressProxy" in nf_type:
            nf_type = nf_type.replace("IngressProxy", "")
        elif "Proxy" in nf_type:
            nf_type = nf_type.replace("Proxy", "")
        hierarchy[city][nf_type].append(proxy)
    return hierarchy

def choose_proxy(data_folder):
    all_proxies = get_all_proxies(data_folder)
    hierarchy = build_proxy_hierarchy(all_proxies)
    cities = sorted(hierarchy.keys())
    print("Available Cities (Supercore):")
    for idx, city in enumerate(cities, 1):
        print(f"{idx}. {city}")
    city_idx = int(input("Select City (enter number): ")) - 1
    city = cities[city_idx]

    nf_types = sorted(hierarchy[city].keys())
    print("Available NF Types:")
    for idx, nf in enumerate(nf_types, 1):
        print(f"{idx}. {nf}")
    nf_idx = int(input("Select NF Type (enter number): ")) - 1
    nf_type = nf_types[nf_idx]

    proxies = sorted(hierarchy[city][nf_type])
    print("Available Proxies:")
    for idx, proxy in enumerate(proxies, 1):
        print(f"{idx}. {proxy}")
    proxy_idx = int(input("Select Proxy (enter number): ")) - 1
    proxy = proxies[proxy_idx]
    return proxy

def main():
    direction = ""
    while direction.lower() not in ["inbound", "outbound"]:
        direction = input("Select direction (inbound/outbound): ").strip().lower()
    if direction == "inbound":
        data_folder = "individual_proxy_inbound"
        column_hint = "response4xxForwardedCounter"
    else:
        data_folder = "individual_proxy_outbound"
        column_hint = "response4xxReceivedCounter"

    proxy_id = choose_proxy(data_folder)
    column_name = input(f"Enter Column Name (e.g., {column_hint}): ").strip()

    # === Date flexibility ===
    start_date_str = input("Enter start date (YYYY-MM-DD) or leave blank for earliest: ").strip()
    end_date_str = input("Enter end date (YYYY-MM-DD) or leave blank for latest: ").strip()
    start_date = start_date_str if start_date_str else None
    end_date = end_date_str if end_date_str else None

    print(f"Starting pipeline for ProxyId: {proxy_id}, Column: {column_name}")

    excel_file = os.path.join(data_folder, f"{proxy_id}.csv")
    if not os.path.exists(excel_file):
        print(f"Data file for proxy '{proxy_id}' not found at {excel_file}")
        return
    print("Data file found:", excel_file)

    # Create plot directory
    plot_dir = os.path.join("plots_individual", direction)
    os.makedirs(plot_dir, exist_ok=True)

    # Step 2: Anomaly Detection (with plot)
    step2_output = detect_anomalies(excel_file, column_name, plot_dir=plot_dir, start_date=start_date, end_date=end_date)
    print("Anomaly detection completed.")
    # Step 3: Filtering Anomalies
    final_output = f"{proxy_id}_{column_name}_final_filtered.csv"
    filter_anomalies(step2_output, final_output, column_name=column_name)
    print("Anomaly filtering completed.")

    print("\nPipeline Complete.")
    print(f"Proxy Data File: {excel_file}")
    print(f"Filtered Anomaly File: {final_output}")

if __name__ == "__main__":
    main()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time:.2f} seconds")