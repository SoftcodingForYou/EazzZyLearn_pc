# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
EazzZyLearn_pc V2025.06 is a real-time closed-loop neurofeedback system for sleep research that detects deep sleep and triggers memory reactivation cues. The system processes EEG data in real-time to detect slow oscillations during sleep and automatically plays audio cues to enhance memory consolidation. The system features both traditional spectral analysis and advanced Muse machine learning sleep classification.

## Architecture

### Data Flow Pipeline
```
Muse EEG Device → OSC Streaming → Signal Processing → Sleep Classification → Slow Oscillation Detection → Timed Audio Cueing
```

### Core Components
- **Backend**: Master controller inheriting from Receiver, orchestrates all real-time processing
- **Receiver**: OSC server for EEG data input, manages data buffering and threading
- **SignalProcessing**: Filters and extracts frequency bands (delta, slow delta, etc.)
- **Sleep Classification**: Dual-mode sleep/wake staging system
  - **SleepWakeState**: Traditional spectral power ratio classification 
  - **MuseSleepClassification**: Advanced ML-based classification using musetools
- **PredictSlowOscillation**: Detects downstates and predicts optimal stimulation timing
- **Cueing**: Audio cue management and stimulation triggering
- **HandleData**: File I/O, session management, and data persistence

### Language Implementations
- **PythonVersion/**: Primary implementation (production-ready)
- **JavaVersion/**: Android port with equivalent architecture
- **SessionReport/**: Post-session analysis and reporting tools

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
python PythonVersion/main_SLEEP.py      # Sleep study mode
python PythonVersion/main_STUDY.py      # Research study mode  
python PythonVersion/main_DEVELOPERMODE.py  # Development/testing

# Testing OSC connection
python test_osc_connection.py           # Test EEG data reception
python debug_osc_messages.py           # Debug OSC message reception
```

### Configuration
All parameters are configured in `PythonVersion/parameters.py`:
- **Session-specific**: Subject info, output directories, electrode mapping
- **Signal processing**: Buffer lengths, frequency bands, filter parameters  
- **OSC streaming**: IP/port for EEG data reception (currently configured for Muse headband)
- **Sleep staging**: Thresholds for wake/sleep classification, Muse classifier configuration
- **Stimulation**: Refractory periods, cue selection, timing parameters
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
2. **Signal Processing**: Real-time filtering and frequency band extraction
3. **Sleep Staging**: Continuous sleep/wake state evaluation using selected method
4. **Slow Oscillation Detection**: Downstate detection and upstate prediction
5. **Cue Triggering**: Audio stimulation with refractory period management

### OSC Data Reception
- Listens on configurable IP:port for `/eeg` messages at 256Hz
- Handles 4-channel Muse EEG data (TP9, AF7, AF8, TP10) padded to 8 channels
- Optional `/muse_metrics` messages for direct sleep classification
- Signal inversion applied (`* -1`) for proper polarity

### Interactive Control
Real-time GUI controls during execution:
- **Channel Selection**: Dynamic switching between electrodes (1-8)
- **Stimulation Control**: Enable/Pause/Force modes via GUI buttons
- **Live Status**: Visual feedback of current sleep/wake state
- **Safe Exit**: Q key for clean shutdown with data preservation

### Threading Architecture
- **Main thread**: Real-time algorithm execution
- **OSC server thread**: EEG data reception
- **Classification threads**: Sleep/wake staging (traditional method only)
- **Stimulation threads**: Non-blocking audio cue playback
- **Data writing threads**: Periodic file I/O operations

### Signal Processing Pipeline
- **Frequency Bands**: Delta (0.5-4Hz), Slow Delta (0.5-2Hz), Alpha (8-12Hz), Beta (12-40Hz), Gamma (25-45Hz)
- **Filtering**: Butterworth filters (order 3) with notch filtering for line noise (49-51Hz)
- **Multi-buffer Architecture**:
  - Main buffer: 30s (7680 samples @ 256Hz)
  - Delta buffer: 5s for slow oscillation detection
  - Sleep buffer: 30s for sleep staging
  - Wake buffer: 3s for rapid wake detection

## Data Output Format
All outputs stored as comma-separated text files with consistent headers:
- **`*_eeg.txt`**: Raw multi-channel EEG signals with timestamps
- **`*_stage.txt`**: Sleep/wake staging decisions with method identification
- **`*_pred.txt`**: Detected downstates and predicted upstates
- **`*_stim.txt`**: Stimulation events, manual controls, and system state changes

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
- **Developer Mode**: Simulated EEG data for algorithm testing
- **Debug Tools**: OSC message monitoring and connection validation
- **Model Testing**: Isolated Muse classifier testing with synthetic data

### Performance Validation
- **Sleep Staging Accuracy**: Traditional method 85.88% validated on research datasets
- **Muse Classifier**: Research-grade accuracy with real-time performance
- **Timing Precision**: Sub-millisecond cue delivery accuracy
- **System Latency**: Total processing latency <50ms for real-time applications

This system represents a state-of-the-art real-time neurofeedback platform specifically designed for sleep research and memory consolidation studies, featuring both traditional signal processing and cutting-edge machine learning approaches for optimal accuracy and reliability.