import streamlit as st
import os
import glob
import time
from datetime import datetime
from collections import defaultdict
import pandas as pd
import multiprocessing
import importlib
from streamlit.components.v1 import html  # Add this import

from ag import detect_anomalies, filter_anomalies_df
from summary import generate_proxy_summary
from anomalyisowithmonthend import detect_anomalies as detect_anomalies_ind
from filteringusingrollingmean import filter_anomalies as filter_anomalies_ind

# Page configuration
st.set_page_config(
    page_title="Proxy Anomaly Detection Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling (remove animation, simplify)
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', sans-serif;
        color: #1a1a1a;
        font-size: 16px;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #003366;
        text-align: center;
        margin-bottom: 1.5rem;
        border-bottom: 2px solid #005580;
        padding-bottom: 0.5rem;
    }
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #004080;
        margin: 1rem 0 0.75rem;
        border-left: 4px solid #0073e6;
        padding-left: 0.5rem;
    }
    .stButton>button {
        border-radius: 8px;
        background-color: #004080;
        color: white;
        padding: 0.5em 1em;
        border: none;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #0066cc;
    }
    .stTabs [role="tab"] {
        font-weight: 500;
        padding: 8px 16px;
        margin-right: 8px;
        border-radius: 6px 6px 0 0;
    }
    .stTabs [role="tab"][aria-selected="true"] {
        background-color: #e6f0ff;
        color: #003366;
        border-bottom: none;
    }
</style>
""", unsafe_allow_html=True)


def get_all_proxies(data_folder):
    """Get all proxy files from the data folder"""
    proxies = []
    for fname in os.listdir(data_folder):
        if fname.endswith('.csv'):
            proxies.append(fname[:-4])
    return proxies


def build_proxy_hierarchy(all_proxies):
    """Build hierarchical structure of proxies by city and NF type"""
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


def process_file_streamlit(args):
    """Process a single file for anomaly detection"""
    file_path, column_name, output_dir, plot_dir, start_date, end_date, iso_params = args
    try:
        anomaly_df = detect_anomalies(
            file_path, column_name, plot_dir=plot_dir, start_date=start_date, end_date=end_date, **iso_params
        )
        # Save in the output_dir, not anomaly_excels
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        final_output = os.path.join(output_dir, f"{base_name}_{column_name}.csv")
        filter_anomalies_df(anomaly_df, final_output, column_name)
        return final_output
    except Exception as e:
        return f"Error processing {file_path}: {e}"


# List of all counters as per the counters file
INBOUND_COUNTERS = [
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

OUTBOUND_COUNTERS = [
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
]


def batch_mode():
    """Batch processing mode for all proxies"""
    st.markdown('<div class="section-header">Batch Mode: Process All Proxies</div>', unsafe_allow_html=True)

    # Use hardcoded counters
    inbound_counters = INBOUND_COUNTERS
    outbound_counters = OUTBOUND_COUNTERS

    # Configuration section
    with st.container():
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Direction**")
            direction = st.selectbox("Select Direction", ["inbound", "outbound"], key="batch_direction")

        with col2:
            st.markdown("**Counter Type**")
            if direction == "inbound":
                counter_options = inbound_counters
            else:
                counter_options = outbound_counters
            counter_choice = st.selectbox("Select Counter Type", counter_options, key="batch_counter")

        with col3:
            st.markdown("**Number of Processes**")
            num_processes = st.number_input(
                "Number of processes", min_value=1, max_value=os.cpu_count(), value=os.cpu_count(), key="batch_num_proc"
            )

    # Isolation Forest parameter tuning
    with st.expander("Isolation Forest Parameters (Advanced)", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators = st.number_input("n_estimators", min_value=1, max_value=500, value=25, step=1, key="batch_n_estimators")
            max_samples = st.number_input("max_samples (fraction)", min_value=0.001, max_value=1.0, value=0.1, step=0.001, format="%.4f", key="batch_max_samples")
        with col2:
            contamination = st.number_input("contamination", min_value=0.000001, max_value=0.5, value=0.000050, step=0.001, format="%.6f", key="batch_contamination")
            max_features = st.number_input("max_features", min_value=0.1, max_value=1.0, value=1.0, step=0.05, format="%.2f", key="batch_max_features")
        with col3:
            bootstrap = st.checkbox("bootstrap", value=True, key="batch_bootstrap")
            random_state = st.number_input("random_state", min_value=0, max_value=9999, value=42, step=1, key="batch_random_state")

    iso_params = dict(
        n_estimators=int(n_estimators),
        max_samples=float(max_samples),
        contamination=float(contamination),
        max_features=float(max_features),
        bootstrap=bootstrap,
        n_jobs=1,
        random_state=int(random_state)
    )

    # Directory and column setup
    if direction == "inbound":
        input_dir = "individual_proxy_inbound"
        output_dir_base = "anomaly_output_inbound"
        plot_dir_base = "anomaly_plots_inbound"
    else:
        input_dir = "individual_proxy_outbound"
        output_dir_base = "anomaly_output_outbound"
        plot_dir_base = "anomaly_plots_outbound"

    column_name = counter_choice
    output_dir = f"{output_dir_base}_{counter_choice}"
    plot_dir = f"{plot_dir_base}_{counter_choice}"

    # Date range selection
    st.markdown("**Date Range (Optional)**")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=None, key="batch_start")
    with col2:
        end_date = st.date_input("End Date", value=None, key="batch_end")

    start_date = start_date.strftime("%Y-%m-%d") if start_date else None
    end_date = end_date.strftime("%Y-%m-%d") if end_date else None

    # File count preview
    if os.path.exists(input_dir):
        file_count = len(glob.glob(os.path.join(input_dir, "*.csv")))
        st.info(f"Found {file_count} files to process in {input_dir}")
    else:
        st.error(f"Input directory {input_dir} not found!")
        return

    # Processing section
    st.markdown("---")
    run_batch = st.button("Run Batch Processing", type="primary", use_container_width=True)

    # --- Always show batch_individual_analysis if batch was run previously ---
    batch_done = st.session_state.get("batch_done", False)
    batch_state = st.session_state.get("batch_state", {})

    if run_batch:
        with st.spinner("Initializing batch processing..."):
            os.makedirs(output_dir, exist_ok=True)
            all_files = glob.glob(os.path.join(input_dir, "*.csv"))
            os.makedirs(plot_dir, exist_ok=True)

            args_list = [
                (file_path, column_name, output_dir, plot_dir, start_date, end_date, iso_params)
                for file_path in all_files
            ]

        # Processing with progress tracking
        start_time = time.time()

        # Create containers for real-time updates
        progress_container = st.container()
        status_container = st.container()

        with progress_container:
            st.markdown("### Processing Progress")
            progress_bar = st.progress(0)
            status_text = st.empty()

        results = []
        success_count = 0

        with multiprocessing.Pool(num_processes) as pool:
            for i, result in enumerate(pool.imap_unordered(process_file_streamlit, args_list)):
                results.append(result)
                if not result.startswith("Error"):
                    success_count += 1

                # Update progress
                progress = (i + 1) / len(args_list)
                progress_bar.progress(progress)
                status_text.text(f"Processed {i + 1}/{len(args_list)} files | Success: {success_count}")

        elapsed = time.time() - start_time

        # Results summary
        with status_container:
            st.markdown("### Processing Results")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Files", len(args_list))
            with col2:
                st.metric("Successful", success_count)
            with col3:
                st.metric("Errors", len(args_list) - success_count)
            with col4:
                st.metric("Time (seconds)", f"{elapsed:.1f}")

        # Success message
        if success_count > 0:
            st.success(f"Successfully processed {success_count} files! Output saved to '{output_dir}'")
            st.info(f"Plots saved in the folder: '{plot_dir}'")

            # Generate summary
            with st.spinner("Generating summary report..."):
                summary_output_file = f"final_summary_{counter_choice}.csv"
                generate_proxy_summary(output_dir, summary_output_file)
                st.success(f"Summary report saved to {summary_output_file}")

                # Show summary preview
                if os.path.exists(summary_output_file):
                    st.markdown("**Summary Preview**")
                    df_summary = pd.read_csv(summary_output_file)
                    st.dataframe(df_summary, use_container_width=True)
                    csv_data = df_summary.to_csv(index=False)
                    st.download_button(
                        label="Download Summary CSV",
                        data=csv_data,
                        file_name=os.path.basename(summary_output_file),
                        mime="text/csv",
                        use_container_width=True,
                        key="batch_summary_download"
                    )

            # --- Store batch state in session_state for persistent navigation ---
            st.session_state["batch_done"] = True
            st.session_state["batch_state"] = {
                "output_dir": output_dir,
                "counter_choice": counter_choice,
                "direction": direction
            }

            batch_individual_analysis(output_dir, counter_choice, direction)

        # Show errors if any
        error_logs = [r for r in results if isinstance(r, str) and r.startswith("Error")]
        if error_logs:
            with st.expander(f"View {len(error_logs)} Error(s)"):
                for err in error_logs:
                    st.error(err)

    # --- If batch was previously run, allow navigation without rerunning batch ---
    elif batch_done and batch_state:
        batch_individual_analysis(
            batch_state["output_dir"],
            batch_state["counter_choice"],
            batch_state["direction"]
        )


def batch_individual_analysis(output_dir, counter_choice, direction):
    st.markdown('<div class="section-header">Analyze Individual Proxy (from Batch Output)</div>', unsafe_allow_html=True)

    # Get all proxies from the batch output directory
    if not os.path.exists(output_dir):
        st.warning(f"No output directory found: {output_dir}")
        return

    # List all CSVs in output_dir, strip .csv and trailing _{counter_choice} if present
    all_proxies = []
    for fname in os.listdir(output_dir):
        if fname.endswith('.csv'):
            proxy_id = fname[:-4]
            # Remove trailing _{counter_choice} if present
            suffix = f"_{counter_choice}"
            if proxy_id.endswith(suffix):
                proxy_id = proxy_id[: -len(suffix)]
            all_proxies.append(proxy_id)

    # Only keep proxies for which the plot file exists
    if direction == "inbound":
        plot_dir = f"anomaly_plots_inbound_{counter_choice}"
    else:
        plot_dir = f"anomaly_plots_outbound_{counter_choice}"

    proxies_with_plot = []
    for proxy_id in all_proxies:
        plot_file_html = os.path.join(plot_dir, f"{proxy_id}_{counter_choice}_plot.html")
        if os.path.exists(plot_file_html):
            proxies_with_plot.append(proxy_id)

    if not proxies_with_plot:
        st.warning("No proxy files with generated plot found for this counter after batch processing.")
        return

    hierarchy = build_proxy_hierarchy(proxies_with_plot)

    st.markdown("**Proxy Selection**")
    col1, col2, col3 = st.columns(3)
    with col1:
        cities = sorted(hierarchy.keys())
        city = st.selectbox("City/Supercore", cities, key="batch_ind_city")
    with col2:
        nf_types = sorted(hierarchy[city].keys())
        nf_type = st.selectbox("NF Type", nf_types, key="batch_ind_nf_type")
    with col3:
        proxies = sorted(hierarchy[city][nf_type])
        proxy_id = st.selectbox("Proxy ID", proxies, key="batch_ind_proxy")

    # File path for selected proxy (add back _{counter_choice})
    proxy_file = os.path.join(output_dir, f"{proxy_id}_{counter_choice}.csv")
    if not os.path.exists(proxy_file):
        st.error(f"Proxy file not found: {proxy_file}")
        return

    # Show preview and download
    st.markdown("---")
    st.markdown("**Anomaly Data Preview**")
    df_results = pd.read_csv(proxy_file)
    st.dataframe(df_results.head(100), use_container_width=True)
    st.metric("Total Anomalies", len(df_results))
    st.metric("Output File", proxy_file)

    csv_data = df_results.to_csv(index=False)
    st.download_button(
        label="Download Results CSV",
        data=csv_data,
        file_name=os.path.basename(proxy_file),
        mime="text/csv",
        use_container_width=True,
        key=f"batch_proxy_download_{proxy_id}"
    )

    # Show plot (guaranteed to exist)
    plot_file_html = os.path.join(plot_dir, f"{proxy_id}_{counter_choice}_plot.html")
    st.markdown("**Anomaly Detection Plot**")
    with open(plot_file_html, "r") as f:
        plot_html = f.read()
    html(plot_html, height=500)


def individual_mode():
    """Individual proxy processing mode"""
    st.markdown('<div class="section-header">Individual Proxy Analysis</div>', unsafe_allow_html=True)

    # Use hardcoded counters
    inbound_counters = INBOUND_COUNTERS
    outbound_counters = OUTBOUND_COUNTERS

    # Configuration section
    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Direction**")
            direction = st.selectbox("Select Direction", ["inbound", "outbound"], key="ind_direction")

        with col2:
            st.markdown("**Column Name**")
            if direction == "inbound":
                data_folder = "individual_proxy_inbound"
                counter_options = inbound_counters
            else:
                data_folder = "individual_proxy_outbound"
                counter_options = outbound_counters
            column_name = st.selectbox("Select column name", counter_options, key="ind_column")

    # Isolation Forest parameter tuning
    with st.expander("Isolation Forest Parameters (Advanced)", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            n_estimators = st.number_input("n_estimators", min_value=1, max_value=500, value=25, step=1, key="ind_n_estimators")
            max_samples = st.number_input("max_samples (fraction)", min_value=0.001, max_value=1.0, value=0.1, step=0.001, format="%.4f", key="ind_max_samples")
        with col2:
            contamination = st.number_input("contamination", min_value=0.000001, max_value=0.5, value=0.000050, step=0.001, format="%.6f", key="ind_contamination")
            max_features = st.number_input("max_features", min_value=0.1, max_value=1.0, value=1.0, step=0.05, format="%.2f", key="ind_max_features")
        with col3:
            bootstrap = st.checkbox("bootstrap", value=True, key="ind_bootstrap")
            random_state = st.number_input("random_state", min_value=0, max_value=9999, value=42, step=1, key="ind_random_state")

    iso_params = dict(
        n_estimators=int(n_estimators),
        max_samples=float(max_samples),
        contamination=float(contamination),
        max_features=float(max_features),
        bootstrap=bootstrap,
        n_jobs=1,
        random_state=int(random_state)
    )

    # Proxy selection
    if not os.path.exists(data_folder):
        st.error(f"Data folder {data_folder} not found!")
        return

    all_proxies = get_all_proxies(data_folder)
    if not all_proxies:
        st.warning(f"No CSV files found in {data_folder}")
        return

    hierarchy = build_proxy_hierarchy(all_proxies)

    st.markdown("**Proxy Selection**")
    col1, col2, col3 = st.columns(3)

    with col1:
        cities = sorted(hierarchy.keys())
        city = st.selectbox("City/Supercore", cities, key="ind_city")

    with col2:
        nf_types = sorted(hierarchy[city].keys())
        nf_type = st.selectbox("NF Type", nf_types, key="ind_nf_type")

    with col3:
        proxies = sorted(hierarchy[city][nf_type])
        proxy_id = st.selectbox("Proxy ID", proxies, key="ind_proxy")

    # Date range
    st.markdown("**Date Range (Optional)**")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date", value=None, key="ind_start")
    with col2:
        end_date = st.date_input("End Date", value=None, key="ind_end")

    start_date = start_date.strftime("%Y-%m-%d") if start_date else None
    end_date = end_date.strftime("%Y-%m-%d") if end_date else None

    # File existence check only (no preview)
    excel_file = os.path.join(data_folder, f"{proxy_id}.csv")
    if os.path.exists(excel_file):
        st.success(f"Data file found: {proxy_id}.csv")
        # Warn if file is very large
        try:
            row_count = sum(1 for _ in open(excel_file, encoding="utf-8")) - 1
            if row_count > 1_000_000:
                st.warning(f"Selected file has {row_count:,} rows. Plotting may be slow or limited to a sample for performance.")
        except Exception:
            pass
    else:
        st.error(f"Data file not found: {excel_file}")
        return

    # Processing
    st.markdown("---")
    if st.button("Start Analysis", type="primary", use_container_width=True):
        plot_dir = os.path.join("plots_individual", direction)
        os.makedirs(plot_dir, exist_ok=True)
        excel_dir = os.path.join("anomaly_excels", direction)
        os.makedirs(excel_dir, exist_ok=True)

        # Processing steps with progress
        progress_container = st.container()

        with progress_container:
            step_progress = st.progress(0)
            step_status = st.empty()

            # Step 1: Anomaly Detection
            step_status.info("Step 1/3: Running anomaly detection...")
            step_progress.progress(0.33)

            try:
                step2_output = detect_anomalies_ind(
                    excel_file, column_name, plot_dir=plot_dir,
                    start_date=start_date, end_date=end_date, **iso_params
                )

                # Step 2: Filtering
                step_status.info("Step 2/3: Filtering anomalies...")
                step_progress.progress(0.66)

                final_output = os.path.join("anomaly_excels", direction, f"{proxy_id}_{column_name}.csv")
                filter_anomalies_ind(step2_output, final_output, column_name=column_name)

                # Step 3: Results
                step_status.info("Step 3/3: Preparing results...")
                step_progress.progress(1.0)

                st.success("Analysis completed successfully!")

                # Results section
                st.markdown("### Results")

                # Show filtered results
                if os.path.exists(final_output):
                    df_results = pd.read_csv(final_output)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Anomalies", len(df_results))
                    with col2:
                        st.metric("Output File", final_output)

                    # Data preview
                    st.markdown("**Anomaly Data Preview**")
                    st.dataframe(df_results.head(100), use_container_width=True)

                    # Download button
                    csv_data = df_results.to_csv(index=False)
                    st.download_button(
                        label="Download Results CSV",
                        data=csv_data,
                        file_name=os.path.basename(final_output),
                        mime="text/csv",
                        use_container_width=True,
                        key=f"ind_proxy_download_{proxy_id}"
                    )

                # Show plot
                plot_file_html = os.path.join(plot_dir, f"{proxy_id}_{column_name}_plot.html")
                if os.path.exists(plot_file_html):
                    st.markdown("**Anomaly Detection Plot**")
                    with open(plot_file_html, "r") as f:
                        plot_html = f.read()
                    html(plot_html, height=500)
                else:
                    st.warning("Plot file not generated")

            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                step_progress.empty()
                step_status.empty()


def run_preprocessing(direction, num_processes):
    # Dynamically import the unified preprocessing module
    preprocess = importlib.import_module("unified_preprocess")
    try:
        result = preprocess.run_preprocessing(direction, num_processes)
        return result
    except Exception as e:
        return f"Error: {e}"


def preprocessing_tab():
    st.markdown('<div class="section-header">Data Preprocessing</div>', unsafe_allow_html=True)
    st.info("Preprocess raw inbound or outbound data files into per-proxy files for anomaly detection.")

    col1, col2 = st.columns(2)
    with col1:
        direction = st.selectbox("Select Data Direction", ["inbound", "outbound"], key="preprocess_direction")
    with col2:
        num_processes = st.number_input(
            "Number of processes", min_value=1, max_value=os.cpu_count(), value=min(8, os.cpu_count()), key="preprocess_num_proc"
        )

    if st.button("Run Preprocessing", type="primary", use_container_width=True):
        with st.spinner(f"Running preprocessing for {direction}..."):
            result = run_preprocessing(direction, int(num_processes))
            if isinstance(result, str) and result.startswith("Error"):
                st.error(result)
            else:
                st.success(f"Preprocessing completed for {direction}.")
                st.info(result)


def main():
    # Header
    st.markdown('<div class="main-header">Proxy Anomaly Detection Dashboard</div>', unsafe_allow_html=True)

    # Navigation bar at the top using tabs (Preprocessing first)
    tabs = st.tabs(["Preprocessing", "Batch Processing", "Individual Analysis"])
    with tabs[0]:
        preprocessing_tab()
    with tabs[1]:
        batch_mode()
    with tabs[2]:
        individual_mode()

if __name__ == "__main__":
    main()