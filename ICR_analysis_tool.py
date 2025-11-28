import sys
import streamlit as st
import math
import os

# Prefer vendor packages (if present) by inserting vendor location at head of sys.path
VENDOR_DIR = os.path.join(os.path.dirname(__file__), 'vendor')
if os.path.isdir(VENDOR_DIR):
    sys.path.insert(0, VENDOR_DIR)

# Attempt to import optional/third-party packages. If any are missing,
# set their variable to None and track the missing package names. This
# allows the app to start and present a clear message to the user when
# a dependency is not installed (instead of crashing on import).
MISSING_DEPENDENCIES = []

try:
    import pandas as pd
except Exception:
    pd = None
    MISSING_DEPENDENCIES.append("pandas")

try:
    import numpy as np
except Exception:
    np = None
    MISSING_DEPENDENCIES.append("numpy")

# Ensure backwards-compatibility alias for older libraries expecting `np.bool8` (not present in NumPy 2.x)
if np is not None and not hasattr(np, 'bool8'):
    try:
        np.bool8 = np.bool_
    except Exception:
        pass

try:
    import nptdms as _nptdms_module
    # Import the class if available; keep module handy for fallbacks.
    try:
        from nptdms import TdmsFile
    except Exception:
        TdmsFile = None
except Exception:
    _nptdms_module = None
    TdmsFile = None
    MISSING_DEPENDENCIES.append("nptdms")


def _open_tdms_file(f, st_ui=None):
    """Module-level helper that opens/reads a TDMS file using the installed/ vendored nptdms API."""
    if TdmsFile is None:
        raise RuntimeError("nptdms package is not available")
    
    file_to_pass = f
    try:
        if hasattr(f, 'read') and hasattr(f, 'seek'):
            f.seek(0)
            file_to_pass = f
    except Exception:
        pass
    
    try:
        if hasattr(TdmsFile, 'read') and callable(getattr(TdmsFile, 'read')):
            return TdmsFile.read(file_to_pass)
    except Exception:
        pass

    try:
        return TdmsFile(file_to_pass)
    except Exception as e:
        if '_nptdms_module' in globals() and _nptdms_module is not None:
            try:
                return _nptdms_module.tdms.read(file_to_pass)
            except Exception:
                pass
        raise


def _get_file_display_name(f):
        """Return a safe, human-readable filename for both path strings and file-like objects.
        - If `f` is a str, return os.path.basename(f)
        - If `f` has a `.name` attribute, return os.path.basename(f.name)
        - Otherwise return 'unknown'
        """
        if isinstance(f, str):
            return os.path.basename(f)
        try:
            name_attr = getattr(f, 'name', None)
            if isinstance(name_attr, str) and name_attr:
                return os.path.basename(name_attr)
        except Exception:
            pass
        return 'unknown'

try:
    from scipy.interpolate import interp1d
except Exception:
    interp1d = None
    MISSING_DEPENDENCIES.append("scipy")

try:
    import plotly.graph_objects as go
    import plotly.express as px
except Exception:
    go = None
    px = None
    MISSING_DEPENDENCIES.append("plotly")


def process_tdms_file(file_object, area_m2, contact_area_cm2, min_pressure_outlier_mpa=0.0, min_resistance_outlier_mohm=0.0, st_ui=None, channel_map=None):
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
    file_name = _get_file_display_name(file_object) or 'unknown_file.tdms'
    try:
        tdms = _open_tdms_file(file_object, st_ui)
        converted_value_group_name = None
        group_names = tdms.groups()  # groups() returns list of strings, not objects
        # Prefer group named 'Converted Value' but try variants; fallback to first group
        for group_name in tdms.groups():
            if group_name.lower() in ['converted value', 'converted values', 'converted_value', 'convertedvalues']:
                converted_value_group_name = group_name
                break
        if converted_value_group_name is None and tdms.groups():
            converted_value_group_name = tdms.groups()[0]

        if converted_value_group_name is None:
           
            return None

        # Detect channel names more flexibly. Get available channel names.
        channels_available = [ch.channel for ch in tdms.group_channels(converted_value_group_name)]

        def find_channel(names, keywords):
            for kw in keywords:
                for n in names:
                    if kw in n.lower():
                        return n
            return None

        # Allow user-provided mapping to override channel selection
        provided_res = None
        provided_force = None
        if channel_map is not None:
            # channel_map keys will be by file base name
            map_key = _get_file_display_name(file_object)
            if map_key in channel_map:
                provided_res = channel_map[map_key].get('res')
                provided_force = channel_map[map_key].get('force')

        res_ch = provided_res or find_channel(channels_available, ['resist', 'resistance', 'ohm', 'mohm', 'mΩ', 'r_', 'r '])
        force_ch = provided_force or find_channel(channels_available, ['force', 'load', 'pressure', 'newton', 'newtons', 'loadcell', 'pn'])

        channel_data = {}
        if res_ch:
            try:
                resistance_obj = next((ch for ch in tdms.group_channels(converted_value_group_name) if ch.channel == res_ch), None)
                if resistance_obj is not None:
                    try:
                        resistance_data = resistance_obj.data
                    except Exception:
                        resistance_data = resistance_obj[:]
                    if resistance_data is not None:
                        channel_data['Resistance'] = resistance_data
            except Exception:
                pass
        
        if force_ch:
            try:
                force_obj = next((ch for ch in tdms.group_channels(converted_value_group_name) if ch.channel == force_ch), None)
                if force_obj is not None:
                    try:
                        force_data = force_obj.data
                    except Exception:
                        force_data = force_obj[:]
                    if force_data is not None:
                        channel_data['Force'] = force_data
            except Exception:
                pass

        need_fallback = False
        if 'Resistance' not in channel_data or 'Force' not in channel_data:
            need_fallback = True

        # Fallback: try to auto-detect numerical channels if the keywords didn't match
        if need_fallback:
            numeric_channels = []
            for ch in tdms.group_channels(converted_value_group_name):
                try:
                    try:
                        vals = ch.data
                    except Exception:
                        vals = ch[:]
                    arr = np.asarray(vals, dtype=float)
                    if arr.size > 1 and not np.all(np.isnan(arr)):
                        numeric_channels.append((ch.channel, arr))
                except Exception:
                    continue
            if len(numeric_channels) >= 2:
                numeric_channels.sort(key=lambda x: np.nanmean(np.abs(x[1])), reverse=True)
                if 'Force' not in channel_data:
                    channel_data['Force'] = numeric_channels[0][1]
                if 'Resistance' not in channel_data:
                    channel_data['Resistance'] = numeric_channels[1][1]
            else:
                return None

        df_current_file = pd.DataFrame(channel_data)
        # Filter out zero-resistance rows (some files may have zero filler values)
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

        # Use SciPy's interp1d if available, otherwise fallback to numpy.interp
        try:
            if interp1d is not None:
                f = interp1d(df_sorted['Pressure'], df_sorted[value_col], kind='linear', fill_value=np.nan, bounds_error=False)
                interp_val = f(common_pressure_range)
            else:
                # numpy.interp will fill outside domain using left/right values - we want NaN instead
                # so we construct an explicit mask for values outside the source range
                src_x = df_sorted['Pressure'].values
                src_y = df_sorted[value_col].values
                # dedupe monotonic x values for numpy.interp
                unique_x, idx_unique = np.unique(src_x, return_index=True)
                unique_y = src_y[idx_unique]
                interp_val = np.interp(common_pressure_range, unique_x, unique_y, left=np.nan, right=np.nan)
        except Exception:
            # If interpolation fails, fallback to creating an array of NaNs for this dataset
            interp_val = np.full_like(common_pressure_range, np.nan, dtype=float)

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


def generate_plot_and_analyze(reference_files, measurement_files, target_pressure, diameter_mm, graph_title, min_pressure_outlier_mpa=0.0, min_resistance_outlier_mohm=0.0, show_debug_tables=False, file_channel_map=None):
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
    # If core third-party libs are missing, return a helpful error string
    missing_core = [m for m in ["pandas", "numpy", "nptdms", "scipy", "plotly"] if m in MISSING_DEPENDENCIES]
    if missing_core:
        missing_str = ", ".join(missing_core)
        msg = (
            f"Missing required Python packages: {missing_str}.\n"
            "Please add them to `requirements.txt` and re-deploy, or run `pip install -r requirements.txt` locally."
        )
        return None, msg

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
        processed_data = process_tdms_file(file_object, area_m2, contact_area_cm2, min_pressure_outlier_mpa, min_resistance_outlier_mohm, st_ui=st, channel_map=file_channel_map)

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
            if show_debug_tables and st is not None:
                try:
                    st.subheader(f"Processed Data for {file_type}")
                    if not df_loading_scan.empty:
                        st.write("Loading scan head:")
                        st.dataframe(df_loading_scan.head())
                    if not df_unloading_scan.empty:
                        st.write("Unloading scan head:")
                        st.dataframe(df_unloading_scan.head())
                except Exception:
                    pass

            if not df_loading_scan.empty:
                reference_loading_dfs_for_averaging.append(df_loading_scan[['Pressure', 'Resistance_mOhm_cm2']].copy())
                min_overall_pressure = min(min_overall_pressure, df_loading_scan['Pressure'].min())
                max_overall_pressure = max(max_overall_pressure, df_loading_scan['Pressure'].max())
            if not df_unloading_scan.empty:
                reference_unloading_dfs_for_averaging.append(df_unloading_scan[['Pressure', 'Resistance_mOhm_cm2']].copy())
                min_overall_pressure = min(min_overall_pressure, df_unloading_scan['Pressure'].min())
                max_overall_pressure = max(max_overall_pressure, df_unloading_scan['Pressure'].max())

        else:
            analysis_results_list.append(f"Reference file '{_get_file_display_name(file_object)}' could not be processed. Check debug output for channel names and errors.")

    for file_object in measurement_files:
        processed_data = process_tdms_file(file_object, area_m2, contact_area_cm2, min_pressure_outlier_mpa, min_resistance_outlier_mohm, st_ui=st, channel_map=file_channel_map)

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
            if show_debug_tables and st is not None:
                try:
                    st.subheader(f"Processed Data for {file_type}")
                    if not df_loading_scan.empty:
                        st.write("Loading scan head:")
                        st.dataframe(df_loading_scan.head())
                    if not df_unloading_scan.empty:
                        st.write("Unloading scan head:")
                        st.dataframe(df_unloading_scan.head())
                except Exception:
                    pass

            if not df_loading_scan.empty:
                measurement_loading_dfs_for_averaging.append(df_loading_scan[['Pressure', 'Resistance_mOhm_cm2']].copy())
                min_overall_pressure = min(min_overall_pressure, df_loading_scan['Pressure'].min())
                max_overall_pressure = max(max_overall_pressure, df_loading_scan['Pressure'].max())
            if not df_unloading_scan.empty:
                measurement_unloading_dfs_for_averaging.append(df_unloading_scan[['Pressure', 'Resistance_mOhm_cm2']].copy())
                min_overall_pressure = min(min_overall_pressure, df_unloading_scan['Pressure'].min())
                max_overall_pressure = max(max_overall_pressure, df_unloading_scan['Pressure'].max())
        else:
            analysis_results_list.append(f"Measurement file '{_get_file_display_name(file_object)}' could not be processed. Check debug output for channel names and errors.")

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
                visible=True,
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
                visible=True,
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


def runtime_check_packages():
    """Attempts to import key packages at runtime and returns a dict with package name, installed status and error message (if any)."""
    packages = ["pandas", "numpy", "nptdms", "scipy", "plotly", "streamlit"]
    results = {}
    for pkg in packages:
        try:
            __import__(pkg)
            results[pkg] = {"ok": True, "error": None}
        except Exception as e:
            results[pkg] = {"ok": False, "error": str(e)}
    return results


def main():
    st.set_page_config(layout="wide")
    st.title("ICR Calculator TDMS files")

    st.info(
        "When outliers are seen change the minimum resistance and pressure in the menu to remove them from the graph, secondly clicking lines in the legend makes them visible. (the standard is only the average lines due to visibility)"
    )

    st.header("Input Parameters")

    # If optional packages are missing, let the user know how to fix it.
    if MISSING_DEPENDENCIES:
        st.error("Missing Python packages: " + ", ".join(MISSING_DEPENDENCIES))
        st.markdown(
            "Please add the missing packages to `requirements.txt` and re-deploy, or install them locally with `pip install -r requirements.txt`."
        )
        st.markdown("If this is a Streamlit Community Cloud deployment, the `requirements.txt` file in the repo will be used for environment installs.")

    # Optionally provide a runtime health-check that checks imports for core packages
    if st.button("Check environment packages"):  # short check available in UI
        with st.spinner("Checking installed packages..."):
            pkg_results = runtime_check_packages()
            ok_pkgs = [p for p, v in pkg_results.items() if v['ok']]
            bad_pkgs = [p for p, v in pkg_results.items() if not v['ok']]
            if ok_pkgs:
                st.success("Installed: " + ", ".join(ok_pkgs))
            if bad_pkgs:
                st.error("Missing/failed imports: " + ", ".join(bad_pkgs))
                for pkg in bad_pkgs:
                    st.write(f"{pkg}: {pkg_results[pkg]['error']}")
                st.markdown("If packages are missing, update `requirements.txt` and re-deploy, or `pip install -r requirements.txt` locally.")

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
    show_debug_tables = st.checkbox("Show processed DataFrames (debug)", value=False, key="show_debug_tables")

    # Allow users to explicitly map channels (Resistance/Force) for each uploaded file if needed
    file_channel_map = {}
    if TdmsFile is not None:
        all_files = (reference_files or []) + (measurement_files or [])
        if all_files:
            st.subheader("Optional: Map Channels for Uploaded TDMS Files")
            for f in all_files:
                try:
                    td = _open_tdms_file(f, st)
                    group_name = None
                    # groups() returns group name strings, not objects
                    for group_str in td.groups():
                        if group_str.lower() in ['converted value', 'converted values', 'converted_value', 'convertedvalues']:
                            group_name = group_str
                            break
                    if group_name is None and td.groups():
                        group_name = td.groups()[0]
                    # Use .channel property to get channel names from TdmsObject instances
                    channel_names = [ch.channel for ch in td.group_channels(group_name)] if group_name is not None else []
                    if not channel_names:
                        st.write(f"No channels detected for {getattr(f, 'name', 'unknown')}")
                        continue
                    col1, col2 = st.columns(2)
                    with col1:
                        res_choice = st.selectbox(f"Resistance channel for {getattr(f, 'name', 'file')}", options=['(auto)'] + channel_names, index=0, key=f"res_{getattr(f,'name','file')}")
                    with col2:
                        force_choice = st.selectbox(f"Force channel for {getattr(f, 'name', 'file')}", options=['(auto)'] + channel_names, index=0, key=f"force_{getattr(f,'name','file')}")
                    chmap = {}
                    if res_choice != '(auto)':
                        chmap['res'] = res_choice
                    if force_choice != '(auto)':
                        chmap['force'] = force_choice
                    if chmap:
                        # Use the same key format as process_tdms_file does when looking it up
                        file_key = _get_file_display_name(f)
                        file_channel_map[file_key] = chmap
                except Exception as e:
                    st.write(f"Could not read file {getattr(f, 'name', 'unknown')} for channel mapping: {e}")
    


    st.header("Analysis Results and Plot")

    plot_placeholder = st.empty()
    results_placeholder = st.empty()

    if generate_button_clicked:
        with st.spinner('Analyzing data and generating plot...'):
        
            # Display all imported data for verification
            st.subheader("Imported Data - Raw Files")
            
            all_files_to_check = (reference_files or []) + (measurement_files or [])
            for file_obj in all_files_to_check:
                try:
                    file_name = _get_file_display_name(file_obj)
                    st.write(f"**File: {file_name}**")
                    
                    # Show file info
                    st.write(f"  File size: {file_obj.size} bytes")
                    
                    processed = process_tdms_file(
                        file_obj, 
                        math.pi * ((diameter_mm/2)/1000)**2,  # area_m2
                        math.pi * ((diameter_mm/2)/10)**2,    # contact_area_cm2
                        min_pressure_outlier_mpa, 
                        min_resistance_outlier_mohm,
                        st_ui=st,
                        channel_map=file_channel_map
                    )
                    
                    if processed:
                        dict_of_scans, file_type = processed
                        df_loading = dict_of_scans.get('loading', pd.DataFrame())
                        df_unloading = dict_of_scans.get('unloading', pd.DataFrame())
                        
                        if not df_loading.empty:
                            st.write(f"  **Loading Scan ({len(df_loading)} points)**")
                            st.dataframe(df_loading.head(10), use_container_width=True)
                        else:
                            st.write("  Loading Scan: No data")
                            
                        if not df_unloading.empty:
                            st.write(f"  **Unloading Scan ({len(df_unloading)} points)**")
                            st.dataframe(df_unloading.head(10), use_container_width=True)
                        else:
                            st.write("  Unloading Scan: No data")
                    else:
                        st.warning(f"Could not process {file_name} - check debug output above for details")
                except Exception as e:
                    import traceback
                    st.error(f"Error displaying data for {file_name}: {e}")
                    st.write(f"```\n{traceback.format_exc()}\n```")

            fig, results_text = generate_plot_and_analyze(
                reference_files, measurement_files, target_pressure, diameter_mm, graph_title,
                min_pressure_outlier_mpa, min_resistance_outlier_mohm,
                show_debug_tables=show_debug_tables,
                file_channel_map=file_channel_map
            )

            if fig is not None:
                plot_placeholder.plotly_chart(fig, use_container_width=True)
            else:
                plot_placeholder.write("No plot could be generated.")

            results_placeholder.text_area("Analysis Results", results_text, height=300)


if __name__ == "__main__":
    main()
