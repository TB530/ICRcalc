import nptdms
import streamlit as st
import pandas as pd
import numpy as np
import math
from nptdms import TdmsFile
from scipy.interpolate import interp1d
import os
import plotly.graph_objects as go
import plotly.express as px


def process_tdms_file(file_object, area_m2, contact_area_cm2, min_pressure_outlier_mpa=0.0, min_resistance_outlier_mohm=0.0):
    """Processes a single TDMS file, extracts all valid data and calculates Resistance_mOhm, Pressure, and Resistance_mOhm_cm2.
       It separates the data into increasing pressure (loading) and decreasing pressure (unloading) segments.
    Args:
        file_object: A file-like object (e.g., from an uploaded file) of the TDMS file.
        area_m2 (float): Contact area in square meters.
        contact_area_cm2 (float): Contact area in square centimeters.
        min_pressure_outlier_mpa (float): Minimum pressure in MPa for outlier removal. Data points below this will be removed.
        min_resistance_outlier_mohm (float): Minimum resistance in mOhm for outlier removal. Data points below this will be removed.
    Returns:
        tuple: (dict_of_scans, file_type) or None on error.
               dict_of_scans contains {'loading': df_loading, 'unloading': df_unloading}
               where each df includes 'Resistance_mOhm', 'Pressure', and 'Resistance_mOhm_cm2'.
               file_type is an extracted label from the filename.
    """
    file_name = getattr(file_object, 'name', 'unknown_file.tdms')
    try:
        tdms = TdmsFile.read(file_object)
        converted_value_group = None
        for g in tdms.groups():
            if g.name == 'Converted Value':
                converted_value_group = g
                break

        if converted_value_group is None:
           
            return None

        channel_data = {}
        for ch in converted_value_group.channels():
            if ch.name in ['Resistance', 'Force']:
                channel_data[ch.name] = ch[:]

        if 'Resistance' not in channel_data or 'Force' not in channel_data:
          
            return None

        df_current_file = pd.DataFrame(channel_data)
        df_filtered = df_current_file[df_current_file['Resistance'] != 0].copy()

        if df_filtered.empty:
            return None

        df_filtered['Resistance_mOhm'] = df_filtered['Resistance'] * 1000
        df_filtered['Pressure'] = df_filtered['Force'] / area_m2 / 1_000_000 
        df_filtered['Resistance_mOhm_cm2'] = df_filtered['Resistance_mOhm'] * contact_area_cm2 

        if min_pressure_outlier_mpa > 0 or min_resistance_outlier_mohm > 0:
            initial_len = len(df_filtered)
            df_filtered = df_filtered[
                (df_filtered['Pressure'] >= min_pressure_outlier_mpa) & 
                (df_filtered['Resistance_mOhm'] >= min_resistance_outlier_mohm)
            ]
            if len(df_filtered) < initial_len:
                print(f"Removed {initial_len - len(df_filtered)} low value outliers from {file_name}")


        max_force_index_label = df_filtered['Force'].idxmax()

        min_points_for_scan = 2

        df_loading_scan = df_filtered.loc[:max_force_index_label].copy()
        if len(df_loading_scan) < min_points_for_scan:
            df_loading_scan = pd.DataFrame()

        df_unloading_scan = df_filtered.loc[max_force_index_label:].copy()
        if len(df_unloading_scan) < min_points_for_scan:
            df_unloading_scan = pd.DataFrame()

        if df_loading_scan.empty and df_unloading_scan.empty:
           
            return None

        base_name = os.path.basename(file_name).replace('.tdms', '')
        label_parts = [p.lower() for p in base_name.split('_')] 
        extracted_label = base_name.lower() 

        

        return {'loading': df_loading_scan, 'unloading': df_unloading_scan}, extracted_label
    except Exception as e:
     
        return None

def interpolate_and_average_curves_func(dfs_to_average, min_pressure, max_pressure, value_col='Resistance_mOhm_cm2'):
    """Interpolates multiple Pressure-Value curves to a common pressure range and averages them.
    Args:
        dfs_to_average (list): A list of pandas DataFrames, each with 'Pressure' and a specified value_col.
        min_pressure (float): The minimum pressure for the common interpolation range.
        max_pressure (float): The maximum pressure for the common interpolation range.
        value_col (str): The column to average (e.g., 'Resistance_mOhm' or 'Resistance_mOhm_cm2').
    Returns:
        pd.DataFrame: An averaged DataFrame with 'Pressure' and 'Avg_Value' columns, or None if no data.
    """
    if not dfs_to_average:
        return None

    if max_pressure <= min_pressure:
        valid_min_ps = [df['Pressure'].min() for df in dfs_to_average if not df['Pressure'].empty]
        valid_max_ps = [df['Pressure'].max() for df in dfs_to_average if not df['Pressure'].empty]

        if valid_min_ps and valid_max_ps:
            min_pressure_for_linspace = np.mean(valid_min_ps)
            max_pressure_for_linspace = np.mean(valid_max_ps)
            if max_pressure_for_linspace <= min_pressure_for_linspace:
                max_pressure_for_linspace += 0.01
        else:
            min_pressure_for_linspace = 0.0
            max_pressure_for_linspace = 1.0
        common_pressure_range = np.linspace(min_pressure_for_linspace, max_pressure_for_linspace, 200)
    else:
         common_pressure_range = np.linspace(min_pressure, max_pressure, 200)

    interpolated_values = []
    for i, df in enumerate(dfs_to_average):
        min_points_for_interpolation = 2

        if value_col not in df.columns or df['Pressure'].empty or len(df) < min_points_for_interpolation:
            continue

        df_sorted = df.sort_values(by='Pressure')

        if df_sorted['Pressure'].min() > common_pressure_range.max() or df_sorted['Pressure'].max() < common_pressure_range.min():
            continue

        f = interp1d(df_sorted['Pressure'], df_sorted[value_col], kind='linear', fill_value=np.nan, bounds_error=False)
        interp_val = f(common_pressure_range)

        if not np.all(np.isnan(interp_val)):
            interpolated_values.append(interp_val)
        else:
            pass

    if not interpolated_values:
        return None

    interpolated_array = np.array(interpolated_values)

    with np.errstate(invalid='ignore'):
        avg_values = np.nanmean(interpolated_array, axis=0)

    valid_indices = ~np.isnan(avg_values)
    if not np.any(valid_indices):
        return None

    avg_df = pd.DataFrame({
        'Pressure': common_pressure_range[valid_indices],
        f'Avg_{value_col}': avg_values[valid_indices]
    })
    return avg_df

def get_icr_at_target_pressure_func(df, target_pressure_mpa, scan_label=""):
    """Calculates Resistance from a DataFrame with 'Pressure' and 'Resistance_mOhm_cm2' columns at the target pressure.
    Args:
        df (pd.DataFrame): DataFrame with 'Pressure' and 'Resistance_mOhm_cm2' columns.
        target_pressure_mpa (float): The target pressure in MPa.
        scan_label (str): Optional label for debugging/logging, e.g., 'Loading' or 'Unloading'.
    Returns:
        tuple: (resistance_at_target, actual_pressure_at_target) or (None, None) if calculation fails.
    """
    if df is None or df.empty or 'Resistance_mOhm_cm2' not in df.columns or 'Pressure' not in df.columns or df['Pressure'].empty:
        return None, None

    df_sorted = df.sort_values(by='Pressure').reset_index(drop=True)

    min_p, max_p = df_sorted['Pressure'].min(), df_sorted['Pressure'].max()

    if not (min_p <= target_pressure_mpa <= max_p):
        idx_closest = (np.abs(df_sorted['Pressure'].values - target_pressure_mpa)).argmin()
        resistance_at_target = df_sorted.iloc[idx_closest]['Resistance_mOhm_cm2']
        actual_pressure_at_target = df_sorted.iloc[idx_closest]['Pressure']
    else:
        resistance_at_target = np.interp(target_pressure_mpa, df_sorted['Pressure'], df_sorted['Resistance_mOhm_cm2'])
        actual_pressure_at_target = target_pressure_mpa

    return resistance_at_target, actual_pressure_at_target

def get_avg_icr_at_target_pressure_func(avg_df, target_pressure_mpa, scan_label=""):
    """Calculates Resistance from an averaged Pressure-Resistance curve at the target pressure.
    Args:
        avg_df (pd.DataFrame): Averaged DataFrame with 'Pressure' and 'Avg_Resistance_mOhm_cm2' columns.
        target_pressure_mpa (float): The target pressure in MPa.
        scan_label (str): Optional label for debugging/logging, e.g., 'Loading' or 'Unloading'.
    Returns:
        tuple: (avg_resistance, avg_actual_pressure) or (None, None) if calculation fails.
    """
    if avg_df is None or avg_df.empty or 'Avg_Resistance_mOhm_cm2' not in avg_df.columns or 'Pressure' not in avg_df.columns or avg_df['Pressure'].empty:
        return None, None

    min_p, max_p = avg_df['Pressure'].min(), avg_df['Pressure'].max()
    if not (min_p <= target_pressure_mpa <= max_p):
        idx_closest = (np.abs(avg_df['Pressure'].values - target_pressure_mpa)).argmin()
        resistance_at_target = avg_df.iloc[idx_closest]['Avg_Resistance_mOhm_cm2']
        actual_pressure_at_target = avg_df.iloc[idx_closest]['Pressure']
  
    else:
        resistance_at_target = np.interp(target_pressure_mpa, avg_df['Pressure'], avg_df['Avg_Resistance_mOhm_cm2'])
        actual_pressure_at_target = target_pressure_mpa

    return resistance_at_target, actual_pressure_at_target


def generate_plot_and_analyze(reference_files, measurement_files, target_pressure, diameter_mm, graph_title, min_pressure_outlier_mpa=0.0, min_resistance_outlier_mohm=0.0):
    """Orchestrates data loading, processing, averaging, result calculation, and plotting.

    Args:
        reference_files (list): List of Streamlit UploadedFile objects for reference files.
        measurement_files (list): List of Streamlit UploadedFile objects for measurement files.
        target_pressure (float): The target pressure for analysis in MPa.
        diameter_mm (float): The diameter in mm.
        graph_title (str): The title for the generated plot.
        min_pressure_outlier_mpa (float): Minimum pressure in MPa for outlier removal.
        min_resistance_outlier_mohm (float): Minimum resistance in mOhm for outlier removal.

    Returns:
        tuple: (plotly.graph_objects.Figure, str) The generated plot figure and the analysis results as a string.
               Returns (None, "") if no files are provided or processing fails.
    """
    all_selected_files = reference_files + measurement_files
    if not all_selected_files:
        return None, "Please select at least one Reference or Measurement TDMS file."


    radius_cm = (diameter_mm / 2) / 10  
    contact_area_cm2 = math.pi * (radius_cm ** 2)
    radius_m = (diameter_mm / 2) / 1000  
    area_m2 = math.pi * (radius_m ** 2)

    analysis_results_list = []
    analysis_results_list.append(f"Target Pressure: {target_pressure} MPa")
    analysis_results_list.append(f"Diameter: {diameter_mm} mm")
    analysis_results_list.append(f"Calculated Contact Area: {contact_area_cm2:.4f} cm²\n")
    
    if min_pressure_outlier_mpa > 0 or min_resistance_outlier_mohm > 0:
        analysis_results_list.append(f"Data filtered: removed points with Pressure < {min_pressure_outlier_mpa:.1f} MPa or Resistance < {min_resistance_outlier_mohm:.1f} mOhm.")

    reference_loading_dfs_for_averaging = []
    reference_unloading_dfs_for_averaging = []
    measurement_loading_dfs_for_averaging = []
    measurement_unloading_dfs_for_averaging = []

    individual_data_frames_for_plotting = []
    individual_resistance_at_target_results = {}

    min_overall_pressure = float('inf')
    max_overall_pressure = float('-inf')

    for file_object in reference_files:
        processed_data = process_tdms_file(file_object, area_m2, contact_area_cm2, min_pressure_outlier_mpa, min_resistance_outlier_mohm)

        if processed_data:
            dict_of_scans, file_type = processed_data

            df_loading_scan = dict_of_scans['loading']
            loading_resistance, loading_actual_pressure = get_icr_at_target_pressure_func(df_loading_scan, target_pressure, "Loading Scan")

            df_unloading_scan = dict_of_scans['unloading']
            unloading_resistance, unloading_actual_pressure = get_icr_at_target_pressure_func(df_unloading_scan, target_pressure, "Unloading Scan")

            individual_resistance_at_target_results[file_type] = {
                'loading_resistance': loading_resistance,
                'loading_actual_pressure': loading_actual_pressure,
                'unloading_resistance': unloading_resistance,
                'unloading_actual_pressure': unloading_actual_pressure
            }

            individual_data_frames_for_plotting.append((df_loading_scan, df_unloading_scan, file_type))

            if not df_loading_scan.empty:
                reference_loading_dfs_for_averaging.append(df_loading_scan[['Pressure', 'Resistance_mOhm_cm2']].copy())
                min_overall_pressure = min(min_overall_pressure, df_loading_scan['Pressure'].min())
                max_overall_pressure = max(max_overall_pressure, df_loading_scan['Pressure'].max())
            if not df_unloading_scan.empty:
                reference_unloading_dfs_for_averaging.append(df_unloading_scan[['Pressure', 'Resistance_mOhm_cm2']].copy())
                min_overall_pressure = min(min_overall_pressure, df_unloading_scan['Pressure'].min())
                max_overall_pressure = max(max_overall_pressure, df_unloading_scan['Pressure'].max())

    for file_object in measurement_files:
        processed_data = process_tdms_file(file_object, area_m2, contact_area_cm2, min_pressure_outlier_mpa, min_resistance_outlier_mohm)

        if processed_data:
            dict_of_scans, file_type = processed_data

            df_loading_scan = dict_of_scans['loading']
            loading_resistance, loading_actual_pressure = get_icr_at_target_pressure_func(df_loading_scan, target_pressure, "Loading Scan")

            df_unloading_scan = dict_of_scans['unloading']
            unloading_resistance, unloading_actual_pressure = get_icr_at_target_pressure_func(df_unloading_scan, target_pressure, "Unloading Scan")

            individual_resistance_at_target_results[file_type] = {
                'loading_resistance': loading_resistance,
                'loading_actual_pressure': loading_actual_pressure,
                'unloading_resistance': unloading_resistance,
                'unloading_actual_pressure': unloading_actual_pressure
            }

            individual_data_frames_for_plotting.append((df_loading_scan, df_unloading_scan, file_type))

            if not df_loading_scan.empty:
                measurement_loading_dfs_for_averaging.append(df_loading_scan[['Pressure', 'Resistance_mOhm_cm2']].copy())
                min_overall_pressure = min(min_overall_pressure, df_loading_scan['Pressure'].min())
                max_overall_pressure = max(max_overall_pressure, df_loading_scan['Pressure'].max())
            if not df_unloading_scan.empty:
                measurement_unloading_dfs_for_averaging.append(df_unloading_scan[['Pressure', 'Resistance_mOhm_cm2']].copy())
                min_overall_pressure = min(min_overall_pressure, df_unloading_scan['Pressure'].min())
                max_overall_pressure = max(max_overall_pressure, df_unloading_scan['Pressure'].max())

    if min_overall_pressure == float('inf') or max_overall_pressure == float('-inf') or max_overall_pressure <= min_overall_pressure:
        if target_pressure > 0:
            min_overall_pressure = target_pressure * 0.5
            max_overall_pressure = target_pressure * 1.5
        else:
            min_overall_pressure = 0.0
            max_overall_pressure = 1.0
        analysis_results_list.append("Warning: Not enough valid pressure data to establish an overall pressure range for averaging. Using a default range.")

    avg_reference_loading_resistance_df = None
    avg_reference_unloading_resistance_df = None
    avg_measurement_loading_resistance_df = None
    avg_measurement_unloading_resistance_df = None

    avg_ref_loading_resistance = None
    avg_ref_unloading_resistance = None
    avg_meas_loading_resistance = None
    avg_meas_unloading_resistance = None

    if reference_loading_dfs_for_averaging:
        avg_reference_loading_resistance_df = interpolate_and_average_curves_func(reference_loading_dfs_for_averaging, min_overall_pressure, max_overall_pressure, value_col='Resistance_mOhm_cm2')
        if avg_reference_loading_resistance_df is not None and not avg_reference_loading_resistance_df.empty:
            avg_ref_loading_resistance, _ = get_avg_icr_at_target_pressure_func(avg_reference_loading_resistance_df, target_pressure, "Avg Ref Loading Scan")

    if reference_unloading_dfs_for_averaging:
        avg_reference_unloading_resistance_df = interpolate_and_average_curves_func(reference_unloading_dfs_for_averaging, min_overall_pressure, max_overall_pressure, value_col='Resistance_mOhm_cm2')
        if avg_reference_unloading_resistance_df is not None and not avg_reference_unloading_resistance_df.empty:
            avg_ref_unloading_resistance, _ = get_avg_icr_at_target_pressure_func(avg_reference_unloading_resistance_df, target_pressure, "Avg Ref Unloading Scan")

    if measurement_loading_dfs_for_averaging:
        avg_measurement_loading_resistance_df = interpolate_and_average_curves_func(measurement_loading_dfs_for_averaging, min_overall_pressure, max_overall_pressure, value_col='Resistance_mOhm_cm2')
        if avg_measurement_loading_resistance_df is not None and not avg_measurement_loading_resistance_df.empty:
            avg_meas_loading_resistance, _ = get_avg_icr_at_target_pressure_func(avg_measurement_loading_resistance_df, target_pressure, "Avg Meas Loading Scan")

    if measurement_unloading_dfs_for_averaging:
        avg_measurement_unloading_resistance_df = interpolate_and_average_curves_func(measurement_unloading_dfs_for_averaging, min_overall_pressure, max_overall_pressure, value_col='Resistance_mOhm_cm2')
        if avg_measurement_unloading_resistance_df is not None and not avg_measurement_unloading_resistance_df.empty:
            avg_meas_unloading_resistance, _ = get_avg_icr_at_target_pressure_func(avg_measurement_unloading_resistance_df, target_pressure, "Avg Meas Unloading Scan")

    final_summary_results = []

    if avg_meas_loading_resistance is not None and avg_ref_loading_resistance is not None:
        diff_loading_val = (avg_meas_loading_resistance - avg_ref_loading_resistance) / 2
        final_summary_results.append(f"Average loading ICR: {diff_loading_val:.4f} mΩ·cm²")


    if avg_meas_unloading_resistance is not None and avg_ref_unloading_resistance is not None:
        diff_unloading_val = (avg_meas_unloading_resistance - avg_ref_unloading_resistance) / 2
        final_summary_results.append(f"Average Unloading ICR: {diff_unloading_val:.4f} mΩ·cm²")

    overall_hysteresis_change_val = None
    if avg_meas_loading_resistance is not None and avg_ref_loading_resistance is not None and \
       avg_meas_unloading_resistance is not None and avg_ref_unloading_resistance is not None:
            overall_hysteresis_change_val = abs(((avg_meas_unloading_resistance - avg_meas_loading_resistance) - \
                                             (avg_ref_unloading_resistance - avg_ref_loading_resistance)) / 2)
            final_summary_results.append(f"ICR Hysteresis: {overall_hysteresis_change_val:.4f} mΩ·cm²")
    else:
        final_summary_results.append("ICR Hysteresis: N/A (Cannot calculate due to missing average resistances)")


    analysis_results_list.append(f"Analysis Complete!\n\nResults @ {target_pressure:.2f} MPa:")
    if final_summary_results:
        analysis_results_list.extend(final_summary_results)
    else:
        analysis_results_list.append("No summary results available.")

    fig = go.Figure()

    if not individual_data_frames_for_plotting and \
       (avg_reference_loading_resistance_df is None or avg_reference_loading_resistance_df.empty) and \
       (avg_reference_unloading_resistance_df is None or avg_reference_unloading_resistance_df.empty) and \
       (avg_measurement_loading_resistance_df is None or avg_measurement_loading_resistance_df.empty) and \
       (avg_measurement_unloading_resistance_df is None or avg_measurement_unloading_resistance_df.empty):
        fig.add_annotation(
            text="No data to plot after filtering and processing.",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        fig.update_layout(
            title=graph_title,
            xaxis_title='Pressure (MPa)', 
            yaxis_title='Resistance (mΩ·cm²)',
            xaxis=dict(range=[0, 100]),
            yaxis=dict(range=[0, 100])
        )
        return fig, "\n".join(analysis_results_list)

    plotly_colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    colors_idx = 0

    resistance_min_all = float('inf')
    resistance_max_all = float('-inf')
    pressure_min_all = float('inf')
    pressure_max_all = float('-inf')

    for df_loading, df_unloading, label_base in individual_data_frames_for_plotting:
        resistance_info = individual_resistance_at_target_results.get(label_base, {})

        if not df_loading.empty and 'Resistance_mOhm_cm2' in df_loading.columns and 'Pressure' in df_loading.columns:
            color = plotly_colors[colors_idx % len(plotly_colors)]
            colors_idx += 1
            loading_resistance_val = resistance_info.get('loading_resistance')
            loading_resistance_str = f'{loading_resistance_val:.4f} mΩ·cm²' if loading_resistance_val is not None else 'N/A'
            legend_label = f'{label_base} Loading (ICR: {loading_resistance_str})'
            fig.add_trace(go.Scatter(
                x=df_loading['Pressure'], y=df_loading['Resistance_mOhm_cm2'], 
                mode='markers',
                name=legend_label,
                visible='legendonly', 
                marker=dict(color=color, symbol='circle', size=5, opacity=0.6),
                hovertemplate=
                    '<b>%{fullData.name}</b><br>'+ 
                    'Pressure: %{x:.2f} MPa<br>'+
                    'Resistance: %{y:.4f} mΩ·cm²<extra></extra>'
            ))
            resistance_min_all = min(resistance_min_all, df_loading['Resistance_mOhm_cm2'].min())
            resistance_max_all = max(resistance_max_all, df_loading['Resistance_mOhm_cm2'].max())
            pressure_min_all = min(pressure_min_all, df_loading['Pressure'].min())
            pressure_max_all = max(pressure_max_all, df_loading['Pressure'].max())

        if not df_unloading.empty and 'Resistance_mOhm_cm2' in df_unloading.columns and 'Pressure' in df_unloading.columns:
            color = plotly_colors[colors_idx % len(plotly_colors)]
            colors_idx += 1
            unloading_resistance_val = resistance_info.get('unloading_resistance')
            unloading_resistance_str = f'{unloading_resistance_val:.4f} mΩ·cm²' if unloading_resistance_val is not None else 'N/A'
            legend_label = f'{label_base} Unloading (ICR: {unloading_resistance_str})'
            fig.add_trace(go.Scatter(
                x=df_unloading['Pressure'], y=df_unloading['Resistance_mOhm_cm2'], 
                mode='markers',
                name=legend_label,
                visible='legendonly', 
                marker=dict(color=color, symbol='x', size=5, opacity=0.6),
                hovertemplate=
                    '<b>%{fullData.name}</b><br>'+ 
                    'Pressure: %{x:.2f} MPa<br>'+ 
                    'Resistance: %{y:.4f} mΩ·cm²<extra></extra>'
            ))
            resistance_min_all = min(resistance_min_all, df_unloading['Resistance_mOhm_cm2'].min())
            resistance_max_all = max(resistance_max_all, df_unloading['Resistance_mOhm_cm2'].max())
            pressure_min_all = min(pressure_min_all, df_unloading['Pressure'].min())
            pressure_max_all = max(pressure_max_all, df_unloading['Pressure'].max())

    if avg_reference_loading_resistance_df is not None and not avg_reference_loading_resistance_df.empty:
        color = plotly_colors[colors_idx % len(plotly_colors)]
        colors_idx += 1
        avg_ref_loading_resistance_str = f'{avg_ref_loading_resistance:.4f} mΩ·cm²' if avg_ref_loading_resistance is not None else 'N/A'
        label = f'Avg Ref Loading (ICR: {avg_ref_loading_resistance_str})'
        fig.add_trace(go.Scatter(
            x=avg_reference_loading_resistance_df['Pressure'], y=avg_reference_loading_resistance_df['Avg_Resistance_mOhm_cm2'], 
            mode='lines+markers',
            name=label,
            visible=True, # This makes averages visible when creating the resistance vs pressure graph
            line=dict(color=color, width=2, dash='solid'),
            marker=dict(symbol='square', size=6),
            hovertemplate=
                '<b>%{fullData.name}</b><br>'+ 
                'Pressure: %{x:.2f} MPa<br>'+ 
                'Resistance: %{y:.4f} mΩ·cm²<extra></extra>'
        ))
        resistance_min_all = min(resistance_min_all, avg_reference_loading_resistance_df['Avg_Resistance_mOhm_cm2'].min())
        resistance_max_all = max(resistance_max_all, avg_reference_loading_resistance_df['Avg_Resistance_mOhm_cm2'].max())
        pressure_min_all = min(pressure_min_all, avg_reference_loading_resistance_df['Pressure'].min())
        pressure_max_all = max(pressure_max_all, avg_reference_loading_resistance_df['Pressure'].max())

    if avg_reference_unloading_resistance_df is not None and not avg_reference_unloading_resistance_df.empty:
        color = plotly_colors[colors_idx % len(plotly_colors)]
        colors_idx += 1
        avg_ref_unloading_resistance_str = f'{avg_ref_unloading_resistance:.4f} mΩ·cm²' if avg_ref_unloading_resistance is not None else 'N/A'
        label = f'Avg Ref Unloading (ICR: {avg_ref_unloading_resistance_str})'
        fig.add_trace(go.Scatter(
            x=avg_reference_unloading_resistance_df['Pressure'], y=avg_reference_unloading_resistance_df['Avg_Resistance_mOhm_cm2'],
            mode='lines+markers',
            name=label,
            visible=True,
            line=dict(color=color, width=2, dash='dash'),
            marker=dict(symbol='triangle-up', size=6),
            hovertemplate=
                '<b>%{fullData.name}</b><br>'+ 
                'Pressure: %{x:.2f} MPa<br>'+ 
                'Resistance: %{y:.4f} mΩ·cm²<extra></extra>'
        ))
        resistance_min_all = min(resistance_min_all, avg_reference_unloading_resistance_df['Avg_Resistance_mOhm_cm2'].min())
        resistance_max_all = max(resistance_max_all, avg_reference_unloading_resistance_df['Avg_Resistance_mOhm_cm2'].max())
        pressure_min_all = min(pressure_min_all, avg_reference_unloading_resistance_df['Pressure'].min())
        pressure_max_all = max(pressure_max_all, avg_reference_unloading_resistance_df['Pressure'].max())

    if avg_measurement_loading_resistance_df is not None and not avg_measurement_loading_resistance_df.empty:
        color = plotly_colors[colors_idx % len(plotly_colors)]
        colors_idx += 1
        avg_meas_loading_resistance_str = f'{avg_meas_loading_resistance:.4f} mΩ·cm²' if avg_meas_loading_resistance is not None else 'N/A'
        label = f'Avg Meas Loading (ICR: {avg_meas_loading_resistance_str})'
        fig.add_trace(go.Scatter(
            x=avg_measurement_loading_resistance_df['Pressure'], y=avg_measurement_loading_resistance_df['Avg_Resistance_mOhm_cm2'],
            mode='lines+markers',
            name=label,
            visible=True, 
            line=dict(color=color, width=2, dash='dot'),
            marker=dict(symbol='circle-open', size=6),
            hovertemplate=
                '<b>%{fullData.name}</b><br>'+ 
                'Pressure: %{x:.2f} MPa<br>'+ 
                'Resistance: %{y:.4f} mΩ·cm²<extra></extra>'
        ))
        resistance_min_all = min(resistance_min_all, avg_measurement_loading_resistance_df['Avg_Resistance_mOhm_cm2'].min())
        resistance_max_all = max(resistance_max_all, avg_measurement_loading_resistance_df['Avg_Resistance_mOhm_cm2'].max())
        pressure_min_all = min(pressure_min_all, avg_measurement_loading_resistance_df['Pressure'].min())
        pressure_max_all = max(pressure_max_all, avg_measurement_loading_resistance_df['Pressure'].max())

    if avg_measurement_unloading_resistance_df is not None and not avg_measurement_unloading_resistance_df.empty:
        color = plotly_colors[colors_idx % len(plotly_colors)]
        colors_idx += 1
        avg_meas_unloading_resistance_str = f'{avg_meas_unloading_resistance:.4f} mΩ·cm²' if avg_meas_unloading_resistance is not None else 'N/A'
        label = f'Avg Meas Unloading (ICR: {avg_meas_unloading_resistance_str})'
        fig.add_trace(go.Scatter(
            x=avg_measurement_unloading_resistance_df['Pressure'], y=avg_measurement_unloading_resistance_df['Avg_Resistance_mOhm_cm2'],
            mode='lines+markers',
            name=label,
            visible=True, 
            line=dict(color=color, width=2, dash='dashdot'),
            marker=dict(symbol='x-open', size=6),
            hovertemplate=
                '<b>%{fullData.name}</b><br>'+ 
                'Pressure: %{x:.2f} MPa<br>'+
                'Resistance: %{y:.4f} mΩ·cm²<extra></extra>'
        ))
        resistance_min_all = min(resistance_min_all, avg_measurement_unloading_resistance_df['Avg_Resistance_mOhm_cm2'].min())
        resistance_max_all = max(resistance_max_all, avg_measurement_unloading_resistance_df['Avg_Resistance_mOhm_cm2'].max())
        pressure_min_all = min(pressure_min_all, avg_measurement_unloading_resistance_df['Pressure'].min())
        pressure_max_all = max(pressure_max_all, avg_measurement_unloading_resistance_df['Pressure'].max())


    fig.update_layout(
        title=graph_title,
        xaxis_title='Pressure (MPa)', 
        yaxis_title='Resistance (mΩ·cm²)', 
        xaxis=dict(gridcolor='lightgrey', griddash='dot', showgrid=True, zeroline=False),
        yaxis=dict(gridcolor='lightgrey', griddash='dot', showgrid=True, zeroline=False),
        legend_title=f'Analysis @ {target_pressure:.2f} MPa',
        legend=dict(
            x=1.05,
            y=1,
            xanchor='left',
            yanchor='top',
            font=dict(size=10)
        ),
        hovermode="x unified", 
        margin=dict(l=50, r=250, t=50, b=50) 
    )

    if np.isfinite(resistance_min_all) and np.isfinite(resistance_max_all) and resistance_min_all != resistance_max_all:
        resistance_range = resistance_max_all - resistance_min_all
        fig.update_yaxes(range=[resistance_min_all - resistance_range * 0.1, resistance_max_all + resistance_range * 0.1])
    elif np.isfinite(resistance_min_all):
        fig.update_yaxes(range=[resistance_min_all * 0.9, resistance_min_all * 1.1]) if resistance_min_all != 0 else fig.update_yaxes(range=[-1, 1]) 
    else:
        fig.update_yaxes(range=[0, 100]) 

    if np.isfinite(pressure_min_all) and np.isfinite(pressure_max_all) and pressure_min_all != pressure_max_all:
        pressure_range = pressure_max_all - pressure_min_all
        fig.update_xaxes(range=[pressure_min_all - pressure_range * 0.1, pressure_max_all + pressure_range * 0.1])
    elif np.isfinite(pressure_min_all):
        fig.update_xaxes(range=[pressure_min_all * 0.9, pressure_min_all * 1.1]) if pressure_min_all != 0 else fig.update_xaxes(range=[-1, 1]) 
    else:
        fig.update_xaxes(range=[max(0, target_pressure * 0.5), target_pressure * 1.5 if target_pressure > 0 else 1.0]) 

    return fig, "\n".join(analysis_results_list)


def main():
    st.set_page_config(layout="wide")
    st.title("ICR Calculator TDMS files")

    st.info(
        "When outliers are seen change the minimum resistance and pressure in the menu to remove them from the graph, secondly clicking lines in the legend makes them visible. (the standard is only the average lines due to visibility)"
    )

    st.header("Input Parameters")

    reference_files = st.file_uploader(
        "Upload Reference TDMS Files (begin, end)",
        type=['tdms'],
        accept_multiple_files=True,
        key="reference_uploader"
    )

    measurement_files = st.file_uploader(
        "Upload Measurement TDMS Files (With PTL, etc.)",
        type=['tdms'],
        accept_multiple_files=True,
        key="measurement_uploader"
    )

    target_pressure = st.number_input(
        "Target Pressure (MPa)",
        min_value=0.0,
        value=1.5,
        step=0.1,
        format="%.2f",
        key="target_pressure_input"
    )

    diameter_mm = st.number_input(
        "Diameter (mm)",
        min_value=0.1,
        value=14.0,
        step=0.1,
        format="%.1f",
        key="diameter_input"
    )

    col1, col2 = st.columns(2)
    with col1:
        min_pressure_outlier_mpa = st.number_input(
            "Minimum Pressure for Outlier Removal (MPa)",
            min_value=0.0,
            value=0.0, 
            step=0.1,
            format="%.2f",
            key="min_pressure_outlier"
        )
    with col2:
        min_resistance_outlier_mohm = st.number_input(
            "Minimum Resistance for Outlier Removal (mΩ·cm²)",
            min_value=0.0,
            value=0.0,
            step=0.1,
            format="%.2f",
            key="min_resistance_outlier"
        )

    graph_title = st.text_input(
        "Graph Title",
        value="Resistance vs. Pressure (Enter name here)",
        key="graph_title_input"
    )

    generate_button_clicked = st.button("Generate Plot and Analyze", key="generate_button")
    


    st.header("Analysis Results and Plot")

    plot_placeholder = st.empty()
    results_placeholder = st.empty()

    if generate_button_clicked:
        with st.spinner('Analyzing data and generating plot...'):
        

            fig, results_text = generate_plot_and_analyze(
                reference_files, measurement_files, target_pressure, diameter_mm, graph_title,
                min_pressure_outlier_mpa, min_resistance_outlier_mohm
            )

            if fig is not None:
                plot_placeholder.plotly_chart(fig, use_container_width=True)
            else:
                plot_placeholder.write("No plot could be generated.")

            results_placeholder.text_area("Analysis Results", results_text, height=300)


if __name__ == "__main__":
    main()
