import pandas as pd
import os
from sklearn.ensemble import IsolationForest
import numpy as np
import plotly.graph_objs as go  # Add this import

def detect_anomalies(file_path, column_name, output_dir=None, plot_dir=None, start_date=None, end_date=None, **iso_params):
    df = pd.read_csv(file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%d-%m-%Y-%H-%M", errors='coerce')
    df.dropna(subset=['Timestamp'], inplace=True)
    # Filter by date range if provided
    if start_date:
        df = df[df['Timestamp'] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df['Timestamp'] <= pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]
    df = df.sort_values(by='Timestamp').reset_index(drop=True)

    # Ensure the counter column exists, if not, create it with zeros
    if column_name not in df.columns:
        df[column_name] = 0

    df['hour'] = df['Timestamp'].dt.hour
    df['day_of_week'] = df['Timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'] >= 5
    df['day'] = df['Timestamp'].dt.day
    df['monthend_flag'] = df['day'].between(26, 30)

    feature_cols = [column_name, 'hour', 'day_of_week', 'is_weekend']
    df_regular = df[~df['monthend_flag']].copy()
    df_monthend = df[df['monthend_flag']].copy()

    # Use passed parameters or defaults
    model = IsolationForest(
        n_estimators=iso_params.get("n_estimators", 25),
        max_samples=iso_params.get("max_samples", 0.1),
        contamination=iso_params.get("contamination", 0.0075),
        max_features=iso_params.get("max_features", 0.80),
        bootstrap=iso_params.get("bootstrap", True),
        n_jobs=iso_params.get("n_jobs", 1),
        random_state=iso_params.get("random_state", 42)
    )

    df_regular['anomaly'] = model.fit_predict(df_regular[feature_cols])
    df_regular['is_anomaly'] = df_regular['anomaly'] == -1

    df_monthend['anomaly'] = model.fit_predict(df_monthend[feature_cols])
    df_monthend['is_anomaly'] = df_monthend['anomaly'] == -1

    df_combined = pd.concat([df_regular, df_monthend]).sort_index().reset_index(drop=True)
    anomaly_df = df_combined[df_combined['is_anomaly']]

    # === PLOT ===
    # Ensure plot_dir is set to anomaly_plots_inbound/outbound_<counter>
    if plot_dir is None:
        if "inbound" in file_path:
            plot_dir = f"anomaly_plots_inbound_{column_name}"
        elif "outbound" in file_path:
            plot_dir = f"anomaly_plots_outbound_{column_name}"
        else:
            plot_dir = f"anomaly_plots_{column_name}"
    else:
        # If plot_dir is just anomaly_plots_inbound or anomaly_plots_outbound, append counter
        if plot_dir.endswith("anomaly_plots_inbound") or plot_dir.endswith("anomaly_plots_outbound"):
            plot_dir = f"{plot_dir}_{column_name}"

    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)
        fig = go.Figure()

        # Downsample main series if too large
        max_points = 10000
        if len(df_combined) > max_points:
            plot_idx = np.linspace(0, len(df_combined) - 1, max_points, dtype=int)
            plot_df = df_combined.iloc[plot_idx]
        else:
            plot_df = df_combined

        # Main time series (downsampled if needed)
        fig.add_trace(go.Scatter(
            x=plot_df['Timestamp'],
            y=plot_df[column_name],
            mode='lines',
            name=column_name,
            line=dict(color='blue'),
            hoverinfo='skip'
        ))

        # Anomalies: red dots and vertical lines to x-axis
        if not anomaly_df.empty:
            # Red dots
            fig.add_trace(go.Scatter(
                x=anomaly_df['Timestamp'],
                y=anomaly_df[column_name],
                mode='markers',
                name='Anomalies',
                marker=dict(color='red', size=8, symbol='circle'),
                hovertemplate="Timestamp: %{x}<br>Value: %{y}<extra></extra>"
            ))
            # Vertical lines to x-axis
            for ts, val in zip(anomaly_df['Timestamp'], anomaly_df[column_name]):
                fig.add_trace(go.Scatter(
                    x=[ts, ts],
                    y=[0, val],
                    mode='lines',
                    line=dict(color='red', width=1, dash='dot'),
                    hoverinfo='skip',
                    showlegend=False
                ))

        fig.update_layout(
            showlegend=False,
            margin=dict(l=40, r=20, t=40, b=40),
            xaxis_title="Timestamp",
            yaxis_title=column_name,
            template="simple_white",
            dragmode="zoom"
        )


        proxy_name = os.path.splitext(os.path.basename(file_path))[0]
        plot_file = os.path.join(plot_dir, f"{proxy_name}_{column_name}_plot.html")
        fig.write_html(plot_file, include_plotlyjs="cdn")
        # Inject config for modebar after saving (removes box/lasso select, enables scroll zoom)
        with open(plot_file, "r", encoding="utf-8") as f:
            html_content = f.read()
        config_script = """
<script type="text/javascript">
document.addEventListener("DOMContentLoaded", function() {
    var gd = document.querySelectorAll('.js-plotly-plot');
    if(gd.length > 0 && window.Plotly) {
        Plotly.newPlot(gd[0], gd[0].data, gd[0].layout, {
            displayModeBar: true,
            modeBarButtonsToRemove: ["select2d", "lasso2d"],
            scrollZoom: true
        });
    }
});
</script>
"""
        html_content = html_content.replace("</body>", config_script + "\n</body>")
        with open(plot_file, "w", encoding="utf-8") as f:
            f.write(html_content)

    columns_to_keep = ['Timestamp', 'ProxyId', column_name, 'hour', 'day_of_week', 'is_weekend', 'day', 'monthend_flag', 'anomaly', 'is_anomaly']
    columns_to_keep = [col for col in columns_to_keep if col in anomaly_df.columns]
    return anomaly_df[columns_to_keep]


def filter_anomalies_df(df, output_file, column_name=None):
    # Save filtered anomalies to the directory specified by output_file
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    # Only keep relevant columns, exclude rolling statistics
    filtered_df = df
    if column_name:
        columns_to_keep = ['Timestamp', 'ProxyId', column_name, 'day']
        columns_to_keep = [col for col in columns_to_keep if col in filtered_df.columns]
        filtered_df = filtered_df[columns_to_keep]
        # Change output_file to proxyname_countername.csv format, but keep in the same directory
        base = os.path.splitext(os.path.basename(output_file))[0]
        if "_" in base:
            proxy_name = "_".join(base.split("_")[:-1])
        else:
            proxy_name = base
        output_file = os.path.join(output_dir, f"{proxy_name}_{column_name}.csv")
    filtered_df.to_csv(output_file, index=False)
    print(f"Filtered data saved to '{output_file}'")