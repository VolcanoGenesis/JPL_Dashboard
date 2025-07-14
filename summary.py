import pandas as pd
import os
import glob
import time

# Parameters for burst/plateau detection
TIME_WINDOW_MINUTES = 10  # window size in minutes
BURST_THRESHOLD = 3       # anomalies >= this in window => burst
PLATEAU_THRESHOLD = 10    # anomalies >= this in window => plateau

def classify_bursts_plateaus(df, time_col='Timestamp'):
    df = df.sort_values(time_col)
    times = pd.to_datetime(df[time_col])
    burst_count = 0
    plateau_count = 0
    used = set()
    for i in range(len(times)):
        if i in used:
            continue
        window_start = times.iloc[i]
        window_end = window_start + pd.Timedelta(minutes=TIME_WINDOW_MINUTES)
        in_window = (times >= window_start) & (times < window_end)
        idxs = df.index[in_window].tolist()
        n = len(idxs)
        if n >= PLATEAU_THRESHOLD:
            plateau_count += 1
            used.update(idxs)
        elif n >= BURST_THRESHOLD:
            burst_count += 1
            used.update(idxs)
    return burst_count, plateau_count

def extract_plateaus(df, time_col='Timestamp'):
    df = df.sort_values(time_col)
    times = pd.to_datetime(df[time_col])
    plateaus = []
    used = set()
    i = 0
    while i < len(times):
        if i in used:
            i += 1
            continue
        window_start = times.iloc[i]
        window_end = window_start + pd.Timedelta(minutes=TIME_WINDOW_MINUTES)
        in_window = (times >= window_start) & (times < window_end)
        idxs = df.index[in_window].tolist()
        n = len(idxs)
        if n >= PLATEAU_THRESHOLD:
            plateau_times = times.loc[idxs]
            plateau_start = plateau_times.min()
            plateau_end = plateau_times.max()
            duration = (plateau_end - plateau_start).total_seconds() / 60.0  # duration in minutes
            plateaus.append({
                'plateau_start': plateau_start,
                'plateau_end': plateau_end,
                'duration_minutes': duration,
                'anomaly_count': n
            })
            used.update(idxs)
            i = idxs[-1] + 1
        else:
            i += 1
    return plateaus

def generate_proxy_summary(directory_path, output_file):
    all_data = []
    burst_plateau_data = []
    plateau_details = []
    csv_files = glob.glob(os.path.join(directory_path, '*.csv'))

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df['date'] = pd.to_datetime(df['Timestamp']).dt.date
            grouped = df.groupby(['ProxyId', 'date']).size().reset_index(name='count')
            all_data.append(grouped)

            for (proxy, date), group in df.groupby(['ProxyId', 'date']):
                burst, plateau = classify_bursts_plateaus(group)
                burst_plateau_data.append({
                    'ProxyId': proxy,
                    'date': date,
                    'bursts': burst,
                    'plateaus': plateau
                })


                plateaus = extract_plateaus(group)
                for p in plateaus:
                    plateau_details.append({
                        'ProxyId': proxy,
                        'date': date,
                        'plateau_start': p['plateau_start'],
                        'plateau_end': p['plateau_end'],
                        'duration_minutes': p['duration_minutes'],
                        'anomaly_count': p['anomaly_count']
                    })

        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue

    if not all_data:
        print("No valid data found in any files")
        return

    combined = pd.concat(all_data, ignore_index=True)
    burst_plateau_df = pd.DataFrame(burst_plateau_data)

    final_grouped = combined.groupby(['ProxyId', 'date'])['count'].sum().reset_index()
    summary_df = final_grouped.pivot(index='ProxyId', columns='date', values='count').reset_index()
    summary_df.rename(columns={'ProxyId': 'proxyid'}, inplace=True)
    date_columns = [col for col in summary_df.columns if col != 'proxyid']
    summary_df[date_columns] = summary_df[date_columns].fillna(0).astype(int)

    # Add bursts and plateaus columns
    burst_df = burst_plateau_df.pivot(index='ProxyId', columns='date', values='bursts').reset_index()
    burst_df.rename(columns={'ProxyId': 'proxyid'}, inplace=True)
    burst_df = burst_df.fillna(0).astype({col: int for col in burst_df.columns if col != 'proxyid'})
    burst_df.columns = [f"{col}_bursts" if col != 'proxyid' else col for col in burst_df.columns]

    plateau_df = burst_plateau_df.pivot(index='ProxyId', columns='date', values='plateaus').reset_index()
    plateau_df.rename(columns={'ProxyId': 'proxyid'}, inplace=True)
    plateau_df = plateau_df.fillna(0).astype({col: int for col in plateau_df.columns if col != 'proxyid'})
    plateau_df.columns = [f"{col}_plateaus" if col != 'proxyid' else col for col in plateau_df.columns]

    # Merge all summaries
    merged = summary_df.merge(burst_df, on='proxyid', how='left').merge(plateau_df, on='proxyid', how='left')

    merged.to_csv(output_file, index=False)
    print(f"Summary saved to {output_file}")
    print(f"Total proxies: {len(summary_df)}")
    print(f"Date range: {min(date_columns)} to {max(date_columns)}")

    # Save plateau details to a separate Excel file
    if plateau_details:
        plateau_df = pd.DataFrame(plateau_details)
        plateau_df = plateau_df.sort_values(['ProxyId', 'date', 'plateau_start'])
        plateau_df.to_csv("plateau_details.csv", index=False)
        print("Plateau details saved to plateau_details.csv")