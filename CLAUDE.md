# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview
EazzZyLearn is a real-time closed-loop neurofeedback system for sleep research that detects deep sleep and triggers memory reactivation cues. The system processes EEG data in real-time to detect slow oscillations during sleep and automatically plays audio cues to enhance memory consolidation.

## Architecture

### Data Flow Pipeline
```
EEG Input → OSC Reception → Signal Processing → Sleep/Wake Staging → Slow Oscillation Detection → Cue Stimulation
```

### Core Components
- **Backend**: Master controller inheriting from Receiver, orchestrates all real-time processing
- **Receiver**: OSC server for EEG data input, manages data buffering and threading
- **SignalProcessing**: Filters and extracts frequency bands (delta, slow delta, etc.)
- **SleepWakeState**: Real-time sleep stage classification using spectral power ratios
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

# Dependencies include: spectrum, keyboard, sounddevice, scipy, matplotlib, numpy, python-osc
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
- **Sleep staging**: Thresholds for wake/sleep classification
- **Stimulation**: Refractory periods, cue selection, timing parameters

## Key Architecture Patterns

### Real-time Processing Loop
The system runs a continuous real-time algorithm (`Backend.real_time_algorithm()`) triggered by incoming EEG samples:
1. **Buffer Management**: Sliding window buffer updated with each sample
2. **Signal Processing**: Real-time filtering and frequency band extraction
3. **Sleep Staging**: Continuous sleep/wake state evaluation
4. **Slow Oscillation Detection**: Downstate detection and upstate prediction
5. **Cue Triggering**: Audio stimulation with refractory period management

### OSC Data Reception
- Listens on configurable IP:port for `/eeg` messages at 256Hz
- Handles 4-channel Muse EEG data (TP9, AF7, AF8, TP10) padded to 8 channels
- Signal inversion applied (`* -1`) for proper polarity

### Interactive Control
Real-time keyboard controls during execution:
- **1-8**: Switch electrode channel for slow oscillation detection
- **P**: Pause stimulation (data collection continues)
- **R**: Resume stimulation (default state)
- **F**: Force stimulation (ignore sleep staging)
- **Q**: Quit program safely (prevents data loss)

### Threading Architecture
- **Main thread**: Real-time algorithm execution
- **OSC server thread**: EEG data reception
- **Stimulation threads**: Non-blocking audio cue playback
- **Data writing threads**: Periodic file I/O operations

## Data Output Format
All outputs stored as comma-separated text files with consistent headers:
- **`*_eeg.txt`**: Raw multi-channel EEG signals with timestamps
- **`*_stages.txt`**: Sleep/wake staging decisions over time
- **`*_pred.txt`**: Detected downstates and predicted upstates
- **`*_stim.txt`**: Stimulation events, manual controls, and system state changes

## Critical Implementation Notes

### Signal Processing Requirements
- Sample rate: 256Hz (Muse) or 250Hz (OpenBCI)
- Buffer management requires careful indexing for real-time sliding windows
- Filter orders and frequency bands are research-validated parameters

### Timing Precision
- Millisecond-precision timestamps for all events
- Refractory period enforcement prevents over-stimulation
- CPU timing synchronization critical for accurate cue delivery

### Memory Management
- Continuous buffer updates without memory allocation in real-time loop
- Periodic data writing to prevent memory overflow
- Thread-safe access to shared data structures