# file: anomaly_detection.py
import pandas as pd
from sklearn.ensemble import IsolationForest
import os
import plotly.graph_objs as go  # Add this import
import numpy as np  # Add this import

def detect_anomalies(file_path, column_name, plot_dir=None, start_date=None, end_date=None, **iso_params):
    df = pd.read_csv(file_path)
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format="%d-%m-%Y-%H-%M", errors='coerce')
    df.dropna(subset=['Timestamp'], inplace=True)
    # === Date filtering ===
    if start_date:
        df = df[df['Timestamp'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['Timestamp'] <= pd.to_datetime(end_date)]
    df = df.sort_values(by='Timestamp').reset_index(drop=True)

    # Ensure the counter column exists, if not, create it with zeros
    if column_name not in df.columns:
        df[column_name] = 0

    df['hour'] = df['Timestamp'].dt.hour
    df['day_of_week'] = df['Timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'] >= 5
    df['day'] = df['Timestamp'].dt.day
    df['monthend_flag'] = df['day'].between(26, 30)

    window = 200
    df['rolling_mean'] = df[column_name].rolling(window).mean()
    df['rolling_std'] = df[column_name].rolling(window).std()
    df['z_score'] = (df[column_name] - df['rolling_mean']) / df['rolling_std']
    df['z_score'] = df['z_score'].fillna(0)

    feature_cols = [column_name, 'hour', 'day_of_week', 'is_weekend', 'z_score']

    df_regular = df[~df['monthend_flag']].copy()
    df_monthend = df[df['monthend_flag']].copy()

    # Use passed parameters or defaults
    model = IsolationForest(
        n_estimators=iso_params.get("n_estimators", 25),
        max_samples=iso_params.get("max_samples", 0.0075),
        contamination=iso_params.get("contamination", 0.001),
        max_features=iso_params.get("max_features", 0.8),
        bootstrap=iso_params.get("bootstrap", True),
        n_jobs=iso_params.get("n_jobs", 1),
        random_state=iso_params.get("random_state", 42)
    )

    # Only fit if there is at least one sample
    if not df_regular.empty:
        df_regular['anomaly'] = model.fit_predict(df_regular[feature_cols])
        df_regular['is_anomaly'] = df_regular['anomaly'] == -1
    else:
        df_regular['anomaly'] = pd.Series([False] * len(df_regular), index=df_regular.index)
        df_regular['is_anomaly'] = pd.Series([False] * len(df_regular), index=df_regular.index)

    if not df_monthend.empty:
        df_monthend['anomaly'] = model.fit_predict(df_monthend[feature_cols])
        df_monthend['is_anomaly'] = df_monthend['anomaly'] == -1
    else:
        df_monthend['anomaly'] = pd.Series([False] * len(df_monthend), index=df_monthend.index)
        df_monthend['is_anomaly'] = pd.Series([False] * len(df_monthend), index=df_monthend.index)

    df_combined = pd.concat([df_regular, df_monthend]).sort_index().reset_index(drop=True)

    # Ensure 'is_anomaly' exists and is boolean
    if 'is_anomaly' in df_combined.columns:
        anomaly_df = df_combined[df_combined['is_anomaly'] == True]
    else:
        anomaly_df = pd.DataFrame(columns=df_combined.columns)

    columns_to_keep = ['Timestamp', 'ProxyId', column_name, 'hour', 'day_of_week', 'is_weekend', 'day', 'monthend_flag',
                       'rolling_mean', 'rolling_std', 'z_score', 'anomaly', 'is_anomaly']
    columns_to_keep = [col for col in columns_to_keep if col in anomaly_df.columns]

    anomaly_df = anomaly_df[columns_to_keep]

    # Save anomalies to a subfolder based on direction
    # Determine direction from file_path or plot_dir
    direction = None
    if plot_dir and ("inbound" in plot_dir):
        direction = "inbound"
    elif plot_dir and ("outbound" in plot_dir):
        direction = "outbound"
    else:
        # fallback: try to infer from file_path
        if "inbound" in file_path:
            direction = "inbound"
        elif "outbound" in file_path:
            direction = "outbound"
        else:
            direction = ""  # fallback to root if not found

    excel_dir = os.path.join("anomaly_excels", direction) if direction else "anomaly_excels"
    os.makedirs(excel_dir, exist_ok=True)
    proxy_name = os.path.splitext(os.path.basename(file_path))[0]
    output_file = os.path.join(excel_dir, f"{proxy_name}_{column_name}.csv")
    anomaly_df.to_csv(output_file, index=False)
    print(f"Anomalies saved to '{output_file}'")

    # === PLOT ===
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
        print(f"Interactive plot saved to '{plot_file}'")

    return output_file