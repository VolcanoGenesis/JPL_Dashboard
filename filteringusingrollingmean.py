# file: filter_anomalies.py
import pandas as pd
import numpy as np
import os

def rolling_zscore_filter(df, window=10, zscore_threshold=2.0):
    # No longer used
    return df

def filter_anomalies(input_file, output_file, column_name=None):
    df = pd.read_csv(input_file)
    # Save filtered anomalies to a subfolder if not already in one
    output_dir = os.path.dirname(output_file)
    if not output_dir:
        output_dir = "anomaly_excels"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, output_file)
    elif not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # Only keep relevant columns, exclude rolling statistics
    filtered_df = df
    if column_name:
        columns_to_keep = ['Timestamp', 'ProxyId', column_name, 'day']
        columns_to_keep = [col for col in columns_to_keep if col in filtered_df.columns]
        filtered_df = filtered_df[columns_to_keep]
        # Change output_file to proxyname_countername.csv format
        base = os.path.splitext(os.path.basename(output_file))[0]
        if "_" in base:
            proxy_name = "_".join(base.split("_")[:-1])
        else:
            proxy_name = base

        # Determine direction from input_file or output_file path
        direction = ""
        if "inbound" in input_file or "inbound" in output_file:
            direction = "inbound"
        elif "outbound" in input_file or "outbound" in output_file:
            direction = "outbound"
        excel_dir = os.path.join("anomaly_excels", direction) if direction else "anomaly_excels"
        os.makedirs(excel_dir, exist_ok=True)
        output_file = os.path.join(excel_dir, f"{proxy_name}_{column_name}.csv")
    filtered_df.to_csv(output_file, index=False)
    print(f"Filtered data saved to '{output_file}'")
    return output_file