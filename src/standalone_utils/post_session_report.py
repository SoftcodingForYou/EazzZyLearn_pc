import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg') # Better interactivity
import os
import sys
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.signal_processing import SignalProcessing

repeated_so_prediction_tolerance_window = 100   # ms --> EZL keeps predicting upstates as long as we 
                                                # are in a slow wave's down phase. Therefore, we 
                                                # have multiple writeouts with respect to the same 
                                                # downstate. We remove those duplicates here.
ezl_eeg_path = r'PATH/TO/_eeg.txt'
ezl_pred_path = r'PATH/TO/_pred.txt'
ezl_stim_path = r'PATH/TO/_stim.txt'

plot_raw_signal             = False
plot_delta_signal           = False
plot_stimulation_timeseries = False
plot_detection_accuracy     = False
plot_prediction_accuracy    = False
plot_grand_average_so       = False
plot_time_freq              = False


# Optional parameters (leave empty or set None if not desired)
muse_eeg_path = r'PATH/TO/RAW/EEG.csv'

# ==================================================================================================

SAVE_PATH = os.path.dirname(ezl_pred_path) if ezl_pred_path else '.'

def load_eeg_data(file_path, is_muse_native):
    """Load EEG data from text file using pandas."""
    try:
        if is_muse_native:
            eeg_data = pd.read_csv(file_path)
        else:
            # Load data with pandas, skipping first 3 lines, no header
            # Data starts at line 4 (0-indexed line 3)
            eeg_data = pd.read_csv(file_path, skiprows=3, header=None)
            
            # Assign column names based on Muse CSV format
            column_names = ['ts', 'ch1', 'ch2', 'ch3', 'ch4']
            eeg_data.columns = column_names[:eeg_data.shape[1]]
        
        print(f"Successfully loaded data from: {file_path}")
        print(f"Data shape: {eeg_data.shape}")
        print(f"Columns: {eeg_data.columns.tolist()}")
        print(f"Sample rate: 256 Hz (Muse standard)")
        print(f"First few rows:")
        print(eeg_data.head())
        
        return eeg_data
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def load_stimulation_data(file_path):
    """Load stimulation data (channel switches and sound cues)."""
    import re
    import os
    try:
        # Read the file skipping first 3 lines
        with open(file_path, 'r') as f:
            lines = f.readlines()[3:]  # Skip first 3 lines
        
        # Parse each line
        timestamps = []
        labels = []
        
        for line in lines:
            if line.strip():  # Skip empty lines
                parts = line.strip().split(', ', 1)  # Split only on first comma
                if len(parts) == 2:
                    # First part is timestamp
                    timestamps.append(float(parts[0]))
                    
                    # Second part is the event description
                    event = parts[1]
                    
                    if 'Switched channel to' in event:
                        # Extract channel info (e.g., "4 (TP10)" -> "TP10")
                        match = re.search(r'\(([^)]+)\)', event)
                        if match:
                            label = f"Channel: {match.group(1)}"
                        else:
                            # Fallback to channel number
                            match = re.search(r'to (\d+)', event)
                            label = f"Channel: {match.group(1)}" if match else event
                    else:
                        # It's a sound file - extract just the filename
                        sound_file = event.split()[0]  # Get first word (filename)
                        label = f"Cue: {sound_file}"
                    
                    labels.append(label)
        
        # Create DataFrame
        stim_data = pd.DataFrame({
            'ts': timestamps,
            'label': labels
        })
        
        print(f"Loaded {len(stim_data)} stimulation events from: {file_path}")
        print(f"Event types: {stim_data['label'].str.split(':').str[0].value_counts().to_dict()}")
        print(stim_data.head())
        
        return stim_data
        
    except Exception as e:
        print(f"Error loading stimulation file: {e}")
        return None

def load_prediction_data(file_path):
    """Load prediction data, extracting timestamps from text."""
    import re
    try:
        # Read the file skipping first 3 lines
        with open(file_path, 'r') as f:
            lines = f.readlines()[3:]  # Skip first 3 lines

        # Parse each line
        downstate_times = []
        predicted_times = []

        for line in lines:
            if line.strip():  # Skip empty lines
                parts = line.strip().split(', ')
                if len(parts) == 2:
                    # First part is downstate timestamp
                    downstate_times.append(float(parts[0]))

                    # Second part contains "Predicted upstate at TIMESTAMP"
                    # Extract the number using regex
                    match = re.search(r'[\d.]+$', parts[1])
                    if match:
                        predicted_times.append(float(match.group()))

        # Create DataFrame
        pred_data = pd.DataFrame({
            'downstate_time': downstate_times,
            'predicted_upstate_time': predicted_times
        })

        print(f"Loaded {len(pred_data)} predictions from: {file_path}")
        print(pred_data.head())

        return pred_data

    except Exception as e:
        print(f"Error loading prediction file: {e}")
        return None

def reconstruct_real_time_channel(eeg_data, timestamps, raw_stim, channel_map, initial_channel):
    switched_channel = np.zeros(len(timestamps))
    
    # Get channel switch events
    if raw_stim is not None:
        channel_switches = raw_stim[raw_stim['label'].str.contains('Channel:')].copy()
        channel_switches['channel_name'] = channel_switches['label'].str.replace('Channel: ', '')
        
        if len(channel_switches) > 0:
            # Sort by timestamp to ensure chronological order
            channel_switches = channel_switches.sort_values('ts').reset_index(drop=True)
            
            # Start with initial channel for the first segment
            start_idx = 0
            current_channel_idx = initial_channel
            
            # Process each segment between channel switches
            for _, switch in channel_switches.iterrows():
                # Find the index where this switch occurs
                switch_idx = np.searchsorted(timestamps, switch['ts'])
                
                # Copy the entire segment from the current channel
                if switch_idx > start_idx:
                    switched_channel[start_idx:switch_idx] = eeg_data[current_channel_idx, start_idx:switch_idx]
                    print(f"   Segment [{start_idx}:{switch_idx}] using channel {current_channel_idx}")
                
                # Update channel for next segment
                channel_name = switch['channel_name']
                if channel_name in channel_map:
                    current_channel_idx = channel_map[channel_name]
                    print(f"   Switching to {channel_name} (channel {current_channel_idx}) at time {switch['ts']:.2f} ms")
                
                start_idx = switch_idx
            
            # Handle the final segment after the last switch
            if start_idx < len(timestamps):
                switched_channel[start_idx:] = eeg_data[current_channel_idx, start_idx:]
                print(f"   Final segment [{start_idx}:{len(timestamps)}] using channel {current_channel_idx}")
            
            print(f"Created switched channel signal using {len(channel_switches)} channel switches")
        else:
            # No switches, use initial channel for entire recording
            switched_channel = eeg_data[initial_channel, :]
            print(f"No channel switches found, using initial channel {initial_channel} for entire recording")
    else:
        # No stim file, use initial channel
        switched_channel = eeg_data[initial_channel, :]
        print(f"No stim file provided, using initial channel {initial_channel}")

    # Add switched channel as 5th channel
    eeg_with_switched = np.vstack([eeg_data, switched_channel[np.newaxis, :]])
    print(f"EEG shape with switched channel: {eeg_with_switched.shape}")

    return eeg_with_switched


# Load recording-specific parameters
# --------------------------------------------------------------------------------------------------
print("Extracting header information ...")
with open(ezl_eeg_path, 'r', encoding='utf8') as file:
    lines = file.readlines()
header_line = lines[0] # Header information of the eeg files
recording = json.loads(header_line)
{"date": "04-09-2025_19-45-57", "sample freq": "256", "subject name": "Offline_NON_Inverted", "age": "0", "sex": "Female", "chosencue": "gong", "electrodes": "{\"TP9\": 0, \"AF7\": 1, \"AF8\": 2, \"TP10\": 3}", "used elec": 1, "default threshold": -75, "artifact threshold": -300, "refractory duration": 6}
recording_datetime  = recording_datetime = datetime.strptime(recording['date'], "%d-%m-%Y_%H-%M-%S")
sampling_rate       = int(recording['sample freq'])
subject_age         = int(recording['age'])
subject_cue         = str(recording['chosencue'])
electrode_map       = json.loads(recording['electrodes'])
initial_channel     = int(recording['used elec'])


# Load files into Pandas DataFrames
# --------------------------------------------------------------------------------------------------
raw_ezl = load_eeg_data(ezl_eeg_path, False)
raw_muse = None
if muse_eeg_path is not None and len(muse_eeg_path) > 0:
    raw_muse = load_eeg_data(muse_eeg_path, True)
raw_pred = load_prediction_data(ezl_pred_path)
raw_stim = load_stimulation_data(ezl_stim_path)

timestamps_ezl = raw_ezl['ts'].to_numpy()
eeg_raw_ezl = raw_ezl.to_numpy().T
eeg_raw_ezl = eeg_raw_ezl[1:, :]

# Reconstruct the analyzed channel (including channel switches during the recording)
eeg_raw_ezl = reconstruct_real_time_channel(eeg_raw_ezl, timestamps_ezl, raw_stim, electrode_map, initial_channel)


# Load and prepare signals stored by EazzZyLearn
# --------------------------------------------------------------------------------------------------
sigproc = SignalProcessing()
sigproc.channel = 4 # Reconstructed real-time channel
buffer_length = len(raw_ezl)
sigproc.main_buffer_length         = buffer_length
sigproc.delta_buffer_length        = buffer_length
sigproc.threshold_buffer_length    = buffer_length
sigproc.sleep_buffer_length        = buffer_length
sigproc.wake_buffer_length         = buffer_length

v_wake, v_sleep, v_filtered_delta, v_delta, v_slowdelta = sigproc.master_extract_signal(eeg_raw_ezl)


# Load and prepare signals coming from MuseData CSV
# --------------------------------------------------------------------------------------------------
if muse_eeg_path is not None and len(muse_eeg_path) > 0:
    raw_muse = raw_muse.fillna(method='ffill') # Same method a EazzZyLearn (forward fill)
    timestamps_muse = raw_muse['ts'].to_numpy()
    timestamps_muse = timestamps_muse*1000 # us to ms
    timestamps_muse = timestamps_muse - timestamps_muse[0] # Only in case of offline reocrding
    eeg_raw_muse = raw_muse.to_numpy().T
    eeg_raw_muse = eeg_raw_muse[1:, :]

    sigproc.channel = 3
    buffer_length = len(raw_muse)
    sigproc.main_buffer_length         = buffer_length
    sigproc.delta_buffer_length        = buffer_length
    sigproc.threshold_buffer_length    = buffer_length
    sigproc.sleep_buffer_length        = buffer_length
    sigproc.wake_buffer_length         = buffer_length

    v_wake_muse, v_sleep_muse, v_filtered_delta_muse, v_delta_muse, v_slowdelta_muse = sigproc.master_extract_signal(eeg_raw_muse)


# Plot full range signals against from Muse native and EazzZyLearn each other
# --------------------------------------------------------------------------------------------------
if plot_raw_signal: # Whole range signal
    fig, ax = plt.subplots(figsize=(12, 6))
    # ax.plot(timestamps_ezl, v_sleep, 'g-', alpha=0.2, label='EZL')
    ax.plot(v_sleep, 'g-', alpha=0.2, label='EZL')
    if muse_eeg_path is not None and len(muse_eeg_path) > 0:
        # ax.plot(timestamps_muse, v_sleep_muse, 'r-', alpha=0.2, label='Muse')
        ax.plot(v_sleep_muse, 'r-', alpha=0.2, label='Muse')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title('Whole range signal: 0.1 - 45 Hz; Notch filter: 49 - 61 Hz)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    output_file = os.path.join(SAVE_PATH, 'whole_range_signal.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"Whole range signal plot saved to: {output_file}")
    # plt.show()

if plot_delta_signal:
    # Do the same for delta signal
    fig, ax = plt.subplots(figsize=(12, 6))
    # ax.plot(timestamps_ezl[:v_delta.size], v_delta, 'g-', alpha=0.2, label='Delta')
    ax.plot(v_delta, 'g-', alpha=0.2, label='Delta')
    if muse_eeg_path is not None and len(muse_eeg_path) > 0:
        ax.plot(v_delta_muse, 'r-', alpha=0.2, label='Slow Delta')
        # ax.plot(timestamps_muse[:v_delta_muse.size], v_delta_muse, 'r-', alpha=0.2, label='Slow Delta')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title('Delta signal: 0.5 - 4 Hz)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    output_file = os.path.join(SAVE_PATH, 'delta_signal.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"Delta signal plot saved to: {output_file}")
    # plt.show()


# Remove downstates that are within 100ms of each other
if raw_pred is not None and len(raw_pred) > 0:
    # Sort by downstate time to ensure chronological order
    raw_pred = raw_pred.sort_values('downstate_time').reset_index(drop=True)
    
    # Keep track of indices to keep
    keep_indices = [0]  # Always keep the first one
    last_kept_time = raw_pred.iloc[0]['downstate_time']
    
    for i in range(1, len(raw_pred)):
        current_time = raw_pred.iloc[i]['downstate_time']
        # Check if current downstate is at least X ms after the last kept one
        if (current_time - last_kept_time) >= repeated_so_prediction_tolerance_window:
            keep_indices.append(i)
            last_kept_time = current_time
    
    # Filter the dataframe to keep only non-overlapping detections
    original_count = len(raw_pred)
    raw_pred = raw_pred.iloc[keep_indices].reset_index(drop=True)
    filtered_count = len(raw_pred)
    
    print(f"Filtered overlapping detections: {original_count} -> {filtered_count} (removed {original_count - filtered_count})")
    print(f"Final prediction count: {len(raw_pred)}")


# Plot detection and prediction timestamps along delta signal
# --------------------------------------------------------------------------------------------------
if plot_stimulation_timeseries:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(timestamps_ezl[:v_delta.size], v_delta, 'g-', alpha=0.5, label='Delta')

    # Add scatter points for downstates and predicted upstates
    if raw_pred is not None and len(raw_pred) > 0:
        # Convert timestamps to indices to get delta values at those times
        # Find closest indices in timestamps_ezl for each prediction time
        for _, row in raw_pred.iterrows():
            # Find index for downstate
            downstate_idx = np.argmin(np.abs(timestamps_ezl - row['downstate_time']))
            if downstate_idx < len(v_delta):
                ax.scatter(row['downstate_time'], v_delta[downstate_idx], 
                        color='black', s=30, zorder=5, alpha=0.7)
            
            # Find index for predicted upstate
            upstate_idx = np.argmin(np.abs(timestamps_ezl - row['predicted_upstate_time']))
            if upstate_idx < len(v_delta):
                ax.scatter(row['predicted_upstate_time'], v_delta[upstate_idx], 
                        color='red', s=30, zorder=5, alpha=0.7)
        
        # Add legend entries for scatter points
        ax.scatter([], [], color='black', s=30, label='Detected Downstates')
        ax.scatter([], [], color='red', s=30, label='Predicted Upstates')

    # Add vertical lines for stimulation events
    if raw_stim is not None and len(raw_stim) > 0:
        # Filter for cue events only
        cue_events = raw_stim[raw_stim['label'].str.contains('Cue:')]
        for _, row in cue_events.iterrows():
            if row['ts'] <= timestamps_ezl[-1]:  # Only plot if within time range
                ax.axvline(x=row['ts'], color='blue', alpha=0.3, linestyle='--', linewidth=1)
        
        # Add legend entry for cues
        ax.axvline(x=-999999, color='blue', alpha=0.3, linestyle='--', label='Cue onset')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (µV)')
    ax.set_title('Delta signal (0.5 - 4 Hz) with Slow Oscillation Detection')
    ax.legend()
    ax.grid(True, alpha=0.3)
    output_file = os.path.join(SAVE_PATH, 'slow_wave_and_stimulations.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    print(f"Plot showing timestamps of slow wave features and stimulations saved to: {output_file}")
    plt.show()


# Extract accuracy metrics for detection and prediction of down- and upstates
# --------------------------------------------------------------------------------------------------
window_look_for_downstate = 200 # ms
downstate_window_length = window_look_for_downstate / 1000 * sampling_rate
down_differences = []
if plot_detection_accuracy:
    # For each detected downstate, take the time delay between the real-time detection and the 
    # offline minimum
    if raw_pred is not None and len(raw_pred) > 0:
        # Convert timestamps to indices to get delta values at those times
        # Find closest indices in timestamps_ezl for each prediction time
        for _, row in raw_pred.iterrows():
            # Find index for downstate
            real_time_downstate_idx = np.argmin(np.abs(timestamps_ezl - row['downstate_time']))
            start_idx = round(real_time_downstate_idx - downstate_window_length/2)
            stop_idx = round(real_time_downstate_idx + downstate_window_length/2)
            if stop_idx < len(v_delta) and start_idx >= 0:
                delta_down = v_delta[start_idx:stop_idx]
                times_down = timestamps_ezl[start_idx:stop_idx]
                offline_time_downstate_idx = np.argmin(delta_down)
                if offline_time_downstate_idx > 0 and offline_time_downstate_idx < delta_down.size - 1: # Otherwise, likely an artifact
                    down_diff = timestamps_ezl[real_time_downstate_idx] - times_down[offline_time_downstate_idx]
                    down_differences.append(down_diff) # in ms
                
    # Positive values mean that real-time implementation is detecting downstates later than actual
    # ones. Expected since we mark them at first sample going back up (one sample later than actual)

    # Plot histogram of detection accuracy
    if len(down_differences) > 0:
        down_differences_ms = np.array(down_differences)

        # Create histogram figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histogram
        n_bins = min(30, max(10, len(down_differences_ms) // 5))  # Adaptive bin number
        counts, bins, patches = ax.hist(down_differences_ms, bins=n_bins,
                                       edgecolor='black', alpha=0.7, color='steelblue')

        # Add vertical line at zero (perfect detection)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2,
                  label='Perfect detection (0 ms)')

        # Calculate and display statistics
        mean_diff = np.mean(down_differences_ms)
        median_diff = np.median(down_differences_ms)
        std_diff = np.std(down_differences_ms)

        # Add vertical lines for mean and median
        ax.axvline(x=mean_diff, color='green', linestyle='-', linewidth=2,
                  label=f'Mean: {mean_diff:.1f} ms')
        ax.axvline(x=median_diff, color='orange', linestyle='-', linewidth=2,
                  label=f'Median: {median_diff:.1f} ms')

        # Labels and title
        ax.set_xlabel('Detection Delay (ms)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Downstate Detection Accuracy (n={len(down_differences)} detections)\n' +
                    f'Mean: {mean_diff:.1f}±{std_diff:.1f} ms, Median: {median_diff:.1f} ms',
                    fontsize=14)

        # Add legend
        ax.legend(loc='upper right')

        # Add grid for better readability
        ax.grid(True, alpha=0.3)

        # Add text box with additional statistics
        textstr = f'Total detections: {len(down_differences)}\n'
        textstr += f'Within ±10ms: {np.sum(np.abs(down_differences_ms) <= 10):.0f} ({100*np.sum(np.abs(down_differences_ms) <= 10)/len(down_differences_ms):.1f}%)\n'
        textstr += f'Within ±20ms: {np.sum(np.abs(down_differences_ms) <= 20):.0f} ({100*np.sum(np.abs(down_differences_ms) <= 20)/len(down_differences_ms):.1f}%)\n'
        textstr += f'Within ±50ms: {np.sum(np.abs(down_differences_ms) <= 50):.0f} ({100*np.sum(np.abs(down_differences_ms) <= 50)/len(down_differences_ms):.1f}%)'

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        # Save figure
        output_file = os.path.join(SAVE_PATH, 'downstate_detection_accuracy.png')
        plt.savefig(output_file, bbox_inches='tight', dpi=150)
        print(f"Detection accuracy histogram saved to: {output_file}")

        # plt.show()
    else:
        print("No valid downstate detections found for accuracy analysis")


if plot_prediction_accuracy:
    # Extract accuracy metrics for upstate predictions
    window_look_for_upstate = 800  # ms to look for actual upstate after downstate
    upstate_window_length = window_look_for_upstate / 1000 * sampling_rate
    up_differences = []

    if raw_pred is not None and len(raw_pred) > 0:
        # For each predicted upstate, find the time delay between prediction and actual peak
        for _, row in raw_pred.iterrows():
            # Get predicted upstate time
            predicted_upstate_time = row['predicted_upstate_time']
            detected_downstate_time = row['downstate_time']

            # Find indices
            predicted_upstate_idx = np.argmin(np.abs(timestamps_ezl - predicted_upstate_time))
            detected_downstate_idx = np.argmin(np.abs(timestamps_ezl - detected_downstate_time))

            # Look for actual upstate peak in a window around the predicted upstate
            # Center the search window on the predicted upstate time
            search_start_idx = detected_downstate_idx
            search_end_idx = min(predicted_upstate_idx + int(upstate_window_length/1.5), len(v_slowdelta) - 1)

            if search_end_idx > search_start_idx:
                # Find the maximum (upstate peak) in the slow delta signal around predicted time
                delta_search_window = v_slowdelta[search_start_idx:search_end_idx]
                times_search_window = timestamps_ezl[search_start_idx:search_end_idx]

                # Find the actual upstate (maximum) in the window
                actual_upstate_idx = np.argmax(delta_search_window)

                # Only consider valid upstates (not at edges which might be artifacts)
                if actual_upstate_idx > 0 and actual_upstate_idx < len(delta_search_window) - 1:
                    actual_upstate_time = times_search_window[actual_upstate_idx]
                    up_diff = predicted_upstate_time - actual_upstate_time
                    up_differences.append(up_diff)

    # Plot histogram of upstate prediction accuracy
    if len(up_differences) > 0:
        up_differences_ms = np.array(up_differences)  # Already in ms

        # Create histogram figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histogram
        n_bins = min(30, max(10, len(up_differences_ms) // 5))  # Adaptive bin number
        counts, bins, patches = ax.hist(up_differences_ms, bins=n_bins,
                                       edgecolor='black', alpha=0.7, color='coral')

        # Add vertical line at zero (perfect prediction)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2,
                  label='Perfect prediction (0 ms)')

        # Calculate and display statistics
        mean_diff = np.mean(up_differences_ms)
        median_diff = np.median(up_differences_ms)
        std_diff = np.std(up_differences_ms)

        # Add vertical lines for mean and median
        ax.axvline(x=mean_diff, color='green', linestyle='-', linewidth=2,
                  label=f'Mean: {mean_diff:.1f} ms')
        ax.axvline(x=median_diff, color='orange', linestyle='-', linewidth=2,
                  label=f'Median: {median_diff:.1f} ms')

        # Labels and title
        ax.set_xlabel('Prediction Error (ms)\n(Positive = predicted after actual)', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(f'Upstate Prediction Accuracy (n={len(up_differences)} predictions)\n' +
                    f'Mean: {mean_diff:.1f}±{std_diff:.1f} ms, Median: {median_diff:.1f} ms',
                    fontsize=14)

        # Add legend
        ax.legend(loc='upper right')

        # Add grid for better readability
        ax.grid(True, alpha=0.3)

        # Add text box with additional statistics
        textstr = f'Total predictions: {len(up_differences)}\n'
        textstr += f'Within ±25ms: {np.sum(np.abs(up_differences_ms) <= 25):.0f} ({100*np.sum(np.abs(up_differences_ms) <= 25)/len(up_differences_ms):.1f}%)\n'
        textstr += f'Within ±50ms: {np.sum(np.abs(up_differences_ms) <= 50):.0f} ({100*np.sum(np.abs(up_differences_ms) <= 50)/len(up_differences_ms):.1f}%)\n'
        textstr += f'Within ±100ms: {np.sum(np.abs(up_differences_ms) <= 100):.0f} ({100*np.sum(np.abs(up_differences_ms) <= 100)/len(up_differences_ms):.1f}%)'

        # Add info about early vs late predictions
        early_preds = np.sum(up_differences_ms < 0)
        late_preds = np.sum(up_differences_ms > 0)
        textstr += f'\nEarly predictions: {early_preds} ({100*early_preds/len(up_differences_ms):.1f}%)'
        textstr += f'\nLate predictions: {late_preds} ({100*late_preds/len(up_differences_ms):.1f}%)'

        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)

        # Save figure
        output_file = os.path.join(SAVE_PATH, 'upstate_prediction_accuracy.png')
        plt.savefig(output_file, bbox_inches='tight', dpi=150)
        print(f"Upstate prediction accuracy histogram saved to: {output_file}")

        # plt.show()
    else:
        print("No valid upstate predictions found for accuracy analysis")


# Create grand average of EEG signal centered on downstates
# --------------------------------------------------------------------------------------------------
if raw_pred is not None and len(raw_pred) > 0 and plot_grand_average_so:
    # Parameters for the window
    pre_time = 1.5  # seconds before downstate
    post_time = 2.0  # seconds after downstate
    
    pre_samples = int(pre_time * sampling_rate)
    post_samples = int(post_time * sampling_rate)
    total_samples = pre_samples + post_samples + 1  # +1 for the center point
    
    # Initialize list to store all epochs
    epochs = []
    
    # Extract epochs around each downstate

    for idx, (_, row) in enumerate(raw_pred.iterrows()):
        downstate_time = row['downstate_time']

        # Find the closest index in the timestamps
        downstate_idx = np.argmin(np.abs(timestamps_ezl - downstate_time))
        # Check if we have enough data before and after
        if downstate_idx - pre_samples >= 0 and downstate_idx + post_samples < len(timestamps_ezl):
            # Extract epoch from reconstructed channel (index 4)
            # epoch = eeg_raw_ezl[4, downstate_idx - pre_samples:downstate_idx + post_samples + 1]
            epoch = v_delta[downstate_idx - pre_samples:downstate_idx + post_samples + 1]
            epochs.append(epoch)
    
    if len(epochs) > 0:
        # Convert to numpy array and compute grand average
        epochs_array = np.array(epochs)
        grand_average = np.mean(epochs_array, axis=0)
        std_error = np.std(epochs_array, axis=0) / np.sqrt(len(epochs))
        
        # Create time axis for plotting
        time_axis = np.linspace(-pre_time, post_time, total_samples) * 1000  # Convert to ms
        
        # Plot the grand average
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        # Plot 1: Grand average with confidence interval
        ax1.plot(time_axis, grand_average, 'b-', linewidth=2, label='Grand Average')
        ax1.fill_between(time_axis,
                         grand_average - std_error,
                         grand_average + std_error,
                         alpha=0.3, color='blue', label='±SEM')
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Real-time downstate')
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax1.set_ylabel('Amplitude (µV)')
        ax1.set_title(f'Grand Average of EEG around Slow Oscillation Downstates (n={len(epochs)})')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Calculate y-limits based on confidence interval with some padding
        y_min = np.min(grand_average - std_error)
        y_max = np.max(grand_average + std_error)
        y_range = y_max - y_min
        y_padding = y_range * 3
        ylim = (y_min - y_padding, y_max + y_padding)

        # Plot 2: All individual epochs overlaid
        ax2.plot(time_axis, epochs_array.T, alpha=0.1, color='gray', linewidth=0.5)
        ax2.plot(time_axis, grand_average, 'b-', linewidth=2, label='Grand Average')
        ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Real-time downstate')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Time relative to downstate (ms)')
        ax2.set_ylabel('Amplitude (µV)')
        ax2.set_title('Individual Epochs')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(ylim)
        
        plt.tight_layout()

        output_file = os.path.join(SAVE_PATH, 'grand_average.png')
        plt.savefig(output_file, bbox_inches='tight', dpi=150)

        # plt.show()
        
        # Print statistics
        print(f"\nGrand Average Statistics:")
        print(f"  Total downstates detected: {len(raw_pred)}")
        print(f"  Valid epochs extracted: {len(epochs)}")
        print(f"  Epochs excluded (edge effects): {len(raw_pred) - len(epochs)}")
        print(f"  Mean amplitude at downstate (t=0): {grand_average[pre_samples]:.2f} µV")
        
        # Find and report the minimum (trough) around the downstate
        search_window_ms = 100  # Search ±100ms around downstate for the actual trough
        search_samples = int(search_window_ms / 1000 * sampling_rate)
        search_start = max(0, pre_samples - search_samples)
        search_end = min(len(grand_average), pre_samples + search_samples)
        
        trough_idx = search_start + np.argmin(grand_average[search_start:search_end])
        trough_time = time_axis[trough_idx]
        trough_value = grand_average[trough_idx]
        print(f"  Trough value: {trough_value:.2f} µV at {trough_time:.1f} ms")
    else:
        print("No valid epochs could be extracted (edge effects)")
else:
    print("No predictions available for grand average")


def stim_time_freq(epochs_array, sample_rate=256):
    """
    Compute time-frequency representation using Morlet wavelets (similar to MNE's tfr_morlet).
    
    Args:
        epochs_array: numpy array of shape (n_epochs, n_samples) containing epoched EEG data
        pre_time: pre-stimulus time in seconds (will be made negative if positive)
        post_time: post-stimulus time in seconds
        sample_rate: sampling rate in Hz (default 256)
    """
    # Work directly with the numpy array of epochs
    n_epochs = epochs_array.shape[0]
    n_samples = epochs_array.shape[1] if epochs_array.ndim > 1 else len(epochs_array)
    
    print(f"Processing {n_epochs} epochs with {n_samples} samples each")
    
    # Simple spectrogram approach (no morlet2 needed)
    from scipy import signal as sp
    
    # Initialize list to store normalized spectrograms
    all_spectrograms_normalized = []
    
    # Process each epoch with spectrogram
    for epoch_idx in range(n_epochs):
        if epochs_array.ndim > 1:
            epoch_data = epochs_array[epoch_idx, :]
        else:
            epoch_data = epochs_array if n_epochs == 1 else epochs_array[epoch_idx]
        
        # Compute spectrogram for this epoch
        # Adjusted parameters for better resolution:
        # - Smaller nperseg (64) = better time resolution (~0.25s windows)
        # - Larger nfft (1024) = better frequency resolution
        # - High overlap (56/64 = 87.5%) = smooth time transitions
        f, t, Sxx = sp.spectrogram(
            epoch_data,
            fs=sample_rate,
            window='hann',
            nperseg=64,   # Smaller window = better time resolution (~0.25s at 256Hz)
            noverlap=56,  # 87.5% overlap for smooth visualization
            nfft=1024,    # Larger FFT = better frequency resolution
            scaling='spectrum'
        )
        
        # Z-score normalization for this epoch
        # For each frequency, compute mean and std across all time points
        freq_mean = np.mean(Sxx, axis=1, keepdims=True)  # Mean across time for each frequency
        freq_std = np.std(Sxx, axis=1, keepdims=True)    # Std across time for each frequency
        
        # Apply z-score normalization to this epoch
        Sxx_normalized = (Sxx - freq_mean) / (freq_std + 1e-10)
        
        all_spectrograms_normalized.append(Sxx_normalized)
    
    # Average across normalized epochs
    avg_power = np.mean(all_spectrograms_normalized, axis=0)
    
    # Convert to dB for better visualization (optional)
    avg_power = 10 * np.log10(avg_power + 1e-10)
    
    # Simple plotting
    # -------------------------------------------------------------------------
    # Focus on relevant frequency range
    freq_mask = (f >= 0.5) & (f <= 20)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the time-frequency representation
    im = ax.pcolormesh(
        t,
        f[freq_mask], 
        avg_power[freq_mask, :],
        cmap='jet',
        shading='auto'
    )
    
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'Time-Frequency Analysis (n={n_epochs} epochs)')
    ax.axvline(0, color='white', linestyle='--', alpha=0.7, linewidth=2)
    
    fig.colorbar(im, ax=ax, label='Normalized Power (0-1)')
    
    # Save figure
    output_file = os.path.join(SAVE_PATH, 'time_frequency.png')
    # plt.savefig(output_file, bbox_inches='tight', dpi=150)
    # plt.show()


if plot_time_freq:
    pre_time = 2.0  # seconds before downstate
    post_time = 3.0  # seconds after downstate
    
    pre_samples = int(pre_time * sampling_rate)
    post_samples = int(post_time * sampling_rate)
    total_samples = pre_samples + post_samples + 1  # +1 for the center point
    
    # Initialize list to store all epochs
    whole_range_epochs = []
    
    # Extract epochs around each downstate
    for _, row in raw_pred.iterrows():
        downstate_time = row['downstate_time']
        downstate_idx = np.argmin(np.abs(timestamps_ezl - downstate_time))
        if downstate_idx >= pre_samples and downstate_idx + post_samples < len(timestamps_ezl):
            whole_range_epoch = v_sleep[downstate_idx - pre_samples:downstate_idx + post_samples + 1]
            whole_range_epochs.append(whole_range_epoch)

    stim_time_freq(np.array(whole_range_epochs))
print('Your last chance to set a breakpoint')