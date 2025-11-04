# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
EazzZyLearn_pc V2025.11 is a real-time closed-loop neurofeedback system for sleep research that detects deep sleep and triggers memory reactivation cues. The system processes EEG data in real-time to detect slow oscillations during sleep and automatically plays audio cues to enhance memory consolidation.

### Key Features
- **Adaptive Learning**: Retrospective learning system that continuously improves upstate prediction accuracy by analyzing actual slow oscillation morphology
- **Dual Sleep Classification**: Traditional spectral analysis and advanced Muse machine learning classifier
- **Online Re-referencing**: Real-time channel re-referencing capability for improved signal quality without affecting stored data
- **Interactive GUI**: PyQt5 interface with settings dialog, runtime controls, and real-time visualization
- **Comprehensive Analysis**: Post-session report generation with 9 configurable analysis plots
- **Flexible Architecture**: Modular design supporting multiple EEG devices (Muse, OpenBCI)

## Architecture

### Data Flow Pipeline
```
Muse EEG Device → OSC Streaming → Signal Processing → Sleep Classification → Slow Oscillation Detection → Timed Audio Cueing
```

### Core Components
- **Backend**: Master controller inheriting from Receiver, orchestrates all real-time processing
- **Receiver**: OSC server for EEG data input, manages data buffering and threading
- **SignalProcessing**: Filters and extracts frequency bands (delta, slow delta, etc.) with optional online re-referencing
- **Sleep Classification**: Dual-mode sleep/wake staging system
  - **SleepWakeState**: Traditional spectral power ratio classification
  - **MuseSleepClassification**: Advanced ML-based classification using musetools
- **PredictSlowOscillation**: Detects downstates and predicts optimal stimulation timing with adaptive learning
- **Cueing**: Audio cue management and stimulation triggering
- **HandleData**: File I/O, session management, and data persistence
- **Frontend**: PyQt5 GUI with settings dialog, real-time controls, and channel/reference selection

### Repository Structure
- **src/backend/**: Core processing modules (backend.py, receiver.py, signal_processing.py, predict_slow_oscillation.py, cueing.py, handle_data.py, disk_io.py, sleep_wake_state.py)
- **src/frontend/**: PyQt5 GUI (frontend.py, settings_dialog.py, pyqt_native_plot_widget.py)
- **src/standalone_utils/**: Post-session analysis and debugging tools (post_session_report.py, muse_osc_simulator.py, test_osc_connection.py, debug_osc_messages.py)
- **src/**: Main entry points (main_SLEEP.py, main_STUDY.py, main_DEVELOPERMODE.py, parameters.py)
- **unmaintained/**: Archived code (JavaVersion/, SessionReport/)

## Development Commands

### Setup and Dependencies
```bash
# Install dependencies
pip install -r requirements.txt

# Dependencies include: spectrum, sounddevice, scipy, matplotlib, numpy, python-osc
```

### Running the System
```bash
# Main execution modes
python src/main_SLEEP.py      # Sleep study mode (production)
python src/main_STUDY.py      # Research study mode
python src/main_DEVELOPERMODE.py  # Development/testing with simulated data

# Testing and debugging
python src/standalone_utils/test_osc_connection.py     # Test EEG data reception
python src/standalone_utils/debug_osc_messages.py      # Debug OSC message reception
python src/standalone_utils/muse_osc_simulator.py      # Offline processing simulator
```

### Configuration
All parameters are configured in `src/parameters.py`:
- **Session-specific**: Subject info, output directories, electrode mapping
- **Signal processing**: Buffer lengths, frequency bands, filter parameters
- **Online re-referencing**: `IDX_REF` (-1 for none, or channel index for reference subtraction)
- **OSC streaming**: IP/port for EEG data reception (currently configured for Muse headband)
- **Sleep staging**: Thresholds for wake/sleep classification, Muse classifier configuration
- **Stimulation**: Refractory periods, cue selection, timing parameters
- **Adaptive prediction**: `TROUGH_MULTIPLICATION` initial coefficient (updated during session)
- **Debugging options**: Sound feedback loop, offline mode, signal plotting
- **Muse Sleep Classifier**: Advanced ML configuration options

## Sleep Classification System

### Dual Classification Architecture
The system supports two sleep classification methods:

#### Traditional Method (SleepWakeState)
- **Spectral Analysis**: Power spectral density using Welch's method
- **Band Ratios**: Research-validated thresholds (85.88% accuracy on n=46 datasets)
- **Sleep Thresholds**: DeltaVSBeta (295), DeltaVSGamma (1500), ThetaVSBeta (20)
- **Wake Thresholds**: BetaVSSpindle (0.5), GammaVSAlpha (0.075), Gamma (0.5)
- **Threading**: Separate wake (1s) and sleep (5s) staging intervals

#### Advanced Muse Classifier (MuseSleepClassification)
- **Machine Learning**: Uses pre-trained models from musetools research
- **Feature Pipeline**: Normalization → Multi-taper spectrograms → NMF → Classification
- **Classifier Options**: Logistic Regression, LDA, Gradient Boosting, UMAP
- **Window Analysis**: 6-second windows with 250ms updates
- **Channel Selection**: TP10 (channel 3) - research validated optimal channel
- **Data Validation**: Robust NaN handling and artifact rejection

### Muse Classifier Configuration
```python
# In parameters.py
USE_MUSE_SLEEP_CLASSIFIER = True         # Enable Muse classifier
MUSE_METRIC_MAP = {                      # Sleep stage mappings
    "Wake": 12, "N1": 13, "N2": 14, "N3": 15, "REM": 16
}
```

## Key Architecture Patterns

### Real-time Processing Loop
The system runs a continuous real-time algorithm (`Backend.real_time_algorithm()`) triggered by incoming EEG samples:
1. **Buffer Management**: Sliding window buffer updated with each sample
2. **Sound Feedback Mode Check**: Optional debugging mode with controlled stimulation timing
3. **Signal Processing**: Real-time filtering and frequency band extraction with optional re-referencing
4. **Sleep Staging**: Continuous sleep/wake state evaluation using selected method
5. **Slow Oscillation Detection**: Downstate detection with validation
6. **Adaptive Upstate Prediction**: Personalized timing prediction with retrospective learning
7. **Cue Triggering**: Audio stimulation with refractory period management

### OSC Data Reception
- Listens on configurable IP:port for `/eeg` messages at 256Hz
- Handles 4-channel Muse EEG data (TP9, AF7, AF8, TP10) padded to 8 channels
- Optional `/muse_metrics` messages for direct sleep classification
- Signal inversion applied (`* -1`) for proper polarity

### Interactive Control
Real-time GUI controls during execution:
- **Channel Selection**: Dynamic switching between electrodes via dropdown (changes logged to stim file)
- **Reference Selection**: Online re-referencing via dropdown (affects processing, not storage)
- **Stimulation Control**: Enable/Pause/Force modes via GUI buttons
- **Live Status**: Visual feedback of current sleep/wake state and processing speed
- **Settings Dialog**: Pre-session configuration via File → Session Settings menu
- **Safe Exit**: Window close or Q key for clean shutdown with data preservation

### Threading Architecture
- **Main thread**: Real-time algorithm execution
- **OSC server thread**: EEG data reception
- **Classification threads**: Sleep/wake staging (traditional method only)
- **Stimulation threads**: Non-blocking audio cue playback
- **Data writing threads**: Periodic file I/O operations

### Signal Processing Pipeline
- **Frequency Bands**: Delta (0.5-4Hz), Slow Delta (0.5-2Hz), Alpha (8-12Hz), Beta (12-40Hz), Gamma (25-45Hz)
- **Filtering**: Butterworth filters (order 3) with dual notch filtering for line noise (50Hz and 60Hz)
- **Online Re-referencing**: Optional reference subtraction (`v_raw = v_raw - v_ref`) applied before filtering
- **Multi-buffer Architecture**:
  - Main buffer: 30s (7680 samples @ 256Hz)
  - Delta buffer: 5s for slow oscillation detection
  - Sleep buffer: 30s for sleep staging
  - Wake buffer: 3s for rapid wake detection

### Adaptive Upstate Prediction Learning
The system features a sophisticated retrospective learning algorithm that continuously improves prediction accuracy:

- **Upstate Validation** (`PredictSlowOscillation.upstate_validation()`):
  - Validates actual upstate peaks after detecting downstates
  - Compares real vs predicted timing from downstate to upstate
  - Updates adaptive `trough_multi` coefficient based on actual morphology

- **Learning Mechanism**:
  - Maintains rolling buffer of 100 `trough_multi` coefficients
  - Averages last 3 coefficients for stability
  - Automatically adapts to individual subject's slow oscillation patterns
  - Predictions become progressively more accurate throughout session

- **Validation Criteria** (7-point algorithm validation):
  - High confidence criteria ensure updates only on clean slow oscillations
  - Each upstate processed exactly once to prevent duplicate corrections
  - Stores predicted samples for later comparison against actual timing

- **Personalization**: System learns individual-specific slow oscillation morphology for optimal timing accuracy

## Data Output Format
All outputs stored as comma-separated text files with consistent headers:
- **`*_eeg.txt`**: Raw multi-channel EEG signals with timestamps (unfiltered, direct from hardware)
- **`*_stage.txt`**: Sleep/wake staging decisions with method identification
- **`*_pred.txt`**: Detected downstates and predicted upstates (empty in sound feedback mode)
- **`*_stim.txt`**: Stimulation events, manual controls, channel/reference switching, and system state changes

### Data Storage Details
- **EEG Storage**: `master_write_data()` saves raw, unfiltered EEG data BEFORE any signal processing
- **No Filtering Delay**: Data is stored directly from OSC stream with only NaN interpolation and optional polarity correction
- **Timestamps**: Absolute Unix epoch timestamps in milliseconds (not normalized)
- **Buffered Writing**: Asynchronous background thread writes every 30 seconds (7680 samples @ 256Hz)
- **CSV Format**: Each row contains `timestamp, ch1, ch2, ch3, ch4, ...`

### Enhanced Sleep Staging Output
When using Muse classifier, stage files include:
- Sleep/wake probabilities from ML model
- Classification confidence scores
- Method identification (traditional vs Muse)
- False positive prevention notifications

## Critical Implementation Notes

### Muse Classifier Integration
- **Model Loading**: Automatic loading of pre-trained NMF and classification models
- **Compatibility**: Handles sklearn version differences with graceful fallbacks
- **Data Validation**: Robust NaN detection and invalid data rejection
- **Channel Mapping**: Automatic channel selection based on research best practices
- **Performance**: Real-time processing with minimal computational overhead

### Signal Processing Requirements
- Sample rate: 256Hz (Muse standard)
- Buffer management requires careful indexing for real-time sliding windows
- Filter orders and frequency bands are research-validated parameters
- Muse classifier requires minimum 6-second windows for accurate classification

### Timing Precision
- Millisecond-precision timestamps for all events
- Refractory period enforcement prevents over-stimulation
- CPU timing synchronization critical for accurate cue delivery
- Muse classifier adds <10ms processing latency

### Memory Management
- Continuous buffer updates without memory allocation in real-time loop
- Periodic data writing to prevent memory overflow
- Thread-safe access to shared data structures
- Muse classifier models loaded once at startup for efficiency

### Error Handling and Robustness
- **Data Quality**: Automatic detection and handling of NaN values, artifacts
- **Model Fallbacks**: Graceful degradation if Muse models fail to load
- **Network Resilience**: OSC connection monitoring and reconnection
- **GUI Responsiveness**: Non-blocking operations preserve real-time performance

## Development and Testing

### Offline Development
- **Developer Mode** (`main_DEVELOPERMODE.py`): Simulated EEG data for algorithm testing
- **OSC Simulator** (`muse_osc_simulator.py`): Comprehensive offline processing program (660 lines) for development
- **Debug Tools**:
  - `test_osc_connection.py`: Validates EEG data reception
  - `debug_osc_messages.py`: Monitors incoming OSC messages
- **Model Testing**: Isolated Muse classifier testing with synthetic data
- **Settings Dialog**: Configure debugging options (offline mode, signal plotting, sound feedback loop)

### Debugging Options (parameters.py)
- **`SOUND_FEEDBACK_LOOP`**: Enable fixed-interval audio stimulation for testing sound-EEG feedback
- **`ENABLE_SIGNAL_PLOT`**: Real-time signal visualization widget (experimental)
- **`OFFLINE_MODE`**: Process pre-recorded data without OSC connection
- **`IDX_REF`**: Test different re-referencing schemes during development

### Performance Validation
- **Sleep Staging Accuracy**: Traditional method 85.88% validated on research datasets
- **Muse Classifier**: Research-grade accuracy with real-time performance
- **Timing Precision**: Sub-millisecond cue delivery accuracy
- **System Latency**: Total processing latency <50ms for real-time applications
- **Adaptive Learning**: Prediction accuracy improves progressively during sessions

## Post-Session Analysis

### Session Report Tool (`src/standalone_utils/post_session_report.py`)
Comprehensive offline analysis tool for recorded sessions with configurable plot generation:

#### Available Analysis Plots
Configure which plots to generate by setting flags to `True`:
- **`plot_raw_signal`**: Whole-range signal (0.1-45 Hz) with notch filter
- **`plot_delta_signal`**: Delta band signal (0.5-4 Hz)
- **`plot_stimulation_timeseries`**: Delta signal with overlaid stimulation markers and downstate/upstate detections
- **`plot_detection_accuracy`**: Histogram of downstate detection timing accuracy
- **`plot_prediction_accuracy`**: Histogram of upstate prediction timing accuracy
- **`plot_phase_polar`**: Polar plot of signal phase at predicted upstate times with Rayleigh statistics
- **`plot_grand_average_so`**: Event-related potential around detected downstates (for closed-loop mode)
- **`plot_grand_average_stim`**: Event-related potential around audio stimulations (for sound feedback mode)
- **`plot_time_freq`**: Time-frequency spectrogram around events

#### Key Features
- **Dual Analysis Modes**:
  - Closed-loop mode: Analyzes downstate detections and upstate predictions from `*_pred.txt`
  - Sound feedback mode: Analyzes audio cue events from `*_stim.txt`
- **Signal Reconstruction**: Automatically reconstructs the analyzed channel including mid-session channel switches
- **Accuracy Metrics**: Compares real-time detection/prediction timestamps with offline ground truth
- **Phase Analysis**: Circular statistics (Rayleigh test) for phase-locking validation
- **Grand Averaging**: Event-related potentials with confidence intervals and individual epoch overlays
- **Edge Handling**: Automatic exclusion of epochs too close to recording boundaries
- **Diagnostic Output**: Timestamp alignment validation and event count summaries

#### Usage Example
```python
# Configure file paths
ezl_eeg_path = r'path/to/*_eeg.txt'
ezl_pred_path = r'path/to/*_pred.txt'  # Optional, for closed-loop analysis
ezl_stim_path = r'path/to/*_stim.txt'

# Enable desired plots
plot_grand_average_stim = True  # For sound feedback sessions
plot_detection_accuracy = True  # For closed-loop sessions with predictions

# Run the script
python src/standalone_utils/post_session_report.py
```

#### Output Files
All plots saved to the same directory as input files:
- `whole_range_signal.png`
- `delta_signal.png`
- `slow_wave_and_stimulations.png`
- `downstate_detection_accuracy.png`
- `upstate_prediction_accuracy.png`
- `phase_polar_plot.png`
- `grand_average.png` (downstates)
- `grand_average_stimulations.png` (audio cues)
- `time_frequency.png`

#### Signal Processing for Visualization
- **Lowpass 20 Hz**: Clean signal for 0-10 Hz pattern visualization (recommended for stimulation timeseries)
- **Delta (0.5-4 Hz)**: Standard slow oscillation analysis
- **Slow Delta (0.5-2 Hz)**: For phase analysis and upstate prediction validation

## Key File Locations

### Core Modules
- **Adaptive prediction**: `src/backend/predict_slow_oscillation.py` (442 lines) - Downstate detection and upstate prediction with retrospective learning
- **Signal processing**: `src/backend/signal_processing.py` (277 lines) - Filtering, band extraction, and online re-referencing
- **Backend controller**: `src/backend/backend.py` (222 lines) - Main real-time processing orchestration
- **Data reception**: `src/backend/receiver.py` (308 lines) - OSC server and buffer management
- **Sleep staging**: `src/backend/sleep_wake_state.py` - Traditional spectral analysis classification
- **Audio cueing**: `src/backend/cueing.py` - Stimulation triggering and playback
- **File I/O**: `src/backend/handle_data.py` (313 lines) - Session management and data persistence
- **Disk operations**: `src/backend/disk_io.py` (122 lines) - Threaded writing to prevent data loss

### Frontend
- **Main GUI**: `src/frontend/frontend.py` (415 lines) - PyQt5 interface with channel/reference controls
- **Settings dialog**: `src/frontend/settings_dialog.py` (456 lines) - Pre-session configuration
- **Plot widget**: `src/frontend/pyqt_native_plot_widget.py` (205 lines) - Real-time signal visualization

### Analysis and Debugging Tools
- **Post-session analysis**: `src/standalone_utils/post_session_report.py` (1044 lines) - Comprehensive offline analysis with 9 plot types
- **OSC simulator**: `src/standalone_utils/muse_osc_simulator.py` (660 lines) - Offline processing for development
- **Connection testing**: `src/standalone_utils/test_osc_connection.py` - EEG data reception validation
- **OSC debugging**: `src/standalone_utils/debug_osc_messages.py` - Message monitoring

### Configuration
- **Parameters**: `src/parameters.py` - All system configuration (channels, thresholds, debugging options)

This system represents a state-of-the-art real-time neurofeedback platform specifically designed for sleep research and memory consolidation studies, featuring both traditional signal processing and cutting-edge machine learning approaches for optimal accuracy and reliability. The system combines adaptive personalization with enhanced flexibility for diverse research applications.