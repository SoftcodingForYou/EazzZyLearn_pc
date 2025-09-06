#!/usr/bin/env python3
"""
Offline Muse OSC Stream Simulator
Loads offline EEG data from CSV files and streams it via OSC
to simulate a real Muse device for EazzZyLearn testing.
"""

import numpy as np
import pandas as pd
import time
import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog,
                            QProgressBar, QSlider, QCheckBox, QDialog,
                            QLineEdit, QGridLayout, QMessageBox, QProgressDialog)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QFont
from pythonosc import udp_client
from threading import Event


class StreamingThread(QThread):
    """Thread for streaming data via OSC."""
    progress_update = pyqtSignal(int, str)
    stream_complete = pyqtSignal()
    
    def __init__(self, simulator):
        super().__init__()
        self.simulator = simulator
        self.stop_event = Event()
        
    def run(self):
        """Stream data via OSC in real-time."""
        if self.simulator.data is None or self.simulator.osc_client is None:
            return
            
        self.stop_event.clear()
        start_time = time.time()
        start_index = self.simulator.current_index
        
        while self.simulator.current_index < len(self.simulator.data) and not self.stop_event.is_set():
            # Calculate timing
            elapsed_time = time.time() - start_time
            expected_index = start_index + int(elapsed_time * self.simulator.sample_rate * self.simulator.playback_speed)
            
            # Send samples to catch up to expected index
            while self.simulator.current_index < min(expected_index, len(self.simulator.data)):
                # Get current sample
                sample = self.simulator.data[self.simulator.current_index].copy()
                
                # Send EEG data
                self.simulator.osc_client.send_message("/eeg", sample.tolist())
                
                # Send sleep scores at 1 Hz (once per second = every 256 samples)
                if self.simulator.sleep_scores is not None and self.simulator.current_index < len(self.simulator.sleep_scores):
                    # Only send muse_metrics once per second (every 256 samples at 256Hz)
                    if self.simulator.current_index % 256 == 0:
                        # Create array with at least 17 elements (indices 0-16)
                        muse_metrics = [0.0] * 17
                        
                        # Get the sleep stage for this second
                        sleep_stage_idx = self.simulator.sleep_scores[self.simulator.current_index]
                        
                        # Set probabilities: 1.0 for the active stage, 0.0 for others
                        # Indices 12-16 correspond to Wake, N1, N2, N3, REM
                        if 12 <= sleep_stage_idx <= 16:  # Valid sleep stage indices
                            # First ensure all sleep-related indices are 0.0
                            for i in range(12, 17):
                                muse_metrics[i] = 0.0
                            # Then set the current stage to 1.0
                            muse_metrics[sleep_stage_idx] = 1.0
                        else:
                            # Default to Wake if invalid stage
                            for i in range(12, 17):
                                muse_metrics[i] = 0.0
                            muse_metrics[12] = 1.0  # Wake
                        
                        self.simulator.osc_client.send_message("/muse_metrics", muse_metrics)
                
                self.simulator.current_index += 1
                
                # Update progress every second
                if self.simulator.current_index % 256 == 0:
                    progress = int((self.simulator.current_index / len(self.simulator.data)) * 100)
                    elapsed_samples = self.simulator.current_index
                    total_samples = len(self.simulator.data)
                    elapsed_time = elapsed_samples / self.simulator.sample_rate
                    total_time = total_samples / self.simulator.sample_rate
                    status = f"Streaming: {elapsed_time:.1f}s / {total_time:.1f}s"
                    self.progress_update.emit(progress, status)
            
            # Sleep briefly to avoid busy waiting
            time.sleep(0.001)
        
        if self.simulator.current_index >= len(self.simulator.data):
            self.stream_complete.emit()
    
    def stop(self):
        """Stop the streaming thread."""
        self.stop_event.set()


class OSCSettingsDialog(QDialog):
    """Dialog for configuring OSC settings."""
    
    def __init__(self, parent, ip, port):
        super().__init__(parent)
        self.setWindowTitle("OSC Settings")
        self.setFixedSize(300, 150)
        
        layout = QGridLayout()
        
        # IP Address
        layout.addWidget(QLabel("Target IP:"), 0, 0)
        self.ip_input = QLineEdit(ip)
        layout.addWidget(self.ip_input, 0, 1)
        
        # Port
        layout.addWidget(QLabel("Target Port:"), 1, 0)
        self.port_input = QLineEdit(str(port))
        layout.addWidget(self.port_input, 1, 1)
        
        # Buttons
        button_layout = QHBoxLayout()
        save_button = QPushButton("Save")
        save_button.clicked.connect(self.accept)
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(save_button)
        button_layout.addWidget(cancel_button)
        
        layout.addLayout(button_layout, 2, 0, 1, 2)
        self.setLayout(layout)
    
    def get_settings(self):
        """Get the configured settings."""
        return self.ip_input.text(), self.port_input.text()


class FileLoadingThread(QThread):
    """Thread for loading CSV files without blocking the UI."""
    loading_complete = pyqtSignal(bool, str)
    progress_update = pyqtSignal(str)
    
    def __init__(self, simulator, load_type):
        super().__init__()
        self.simulator = simulator
        self.load_type = load_type  # 'csv' or specific file type
        
    def run(self):
        """Load files in background thread."""
        try:
            if self.load_type == 'csv':
                self.progress_update.emit("Loading EEG data...")
                success = self.simulator.load_csv_files()
                if success:
                    self.loading_complete.emit(True, "CSV files loaded successfully")
                else:
                    self.loading_complete.emit(False, "Failed to load CSV files")
            else:
                self.loading_complete.emit(False, "Unknown load type")
        except Exception as e:
            self.loading_complete.emit(False, str(e))


class MuseOSCSimulator(QMainWindow):
    """Main window for the Muse OSC Stream Simulator."""
    
    def __init__(self):
        super().__init__()
        self.sample_rate = 256  # Muse standard sample rate
        self.target_ip = "127.0.0.1"  # localhost
        self.target_port = 12345  # Default OSC port from parameters
        self.osc_client = None
        
        self.data = None
        self.timestamps = None
        self.sleep_scores = None
        self.current_index = 0
        self.playback_speed = 1.0
        
        self.streaming_thread = None
        
        # File paths
        self.eeg_filepath = None
        self.ss_filepath = None
        
        # Sleep stage mapping from parameters.py
        self.sleep_stage_map = {
            "W": 12,   # Wake
            "Wake": 12,
            "N1": 13,
            "N2": 14,
            "N3": 15,
            "REM": 16
        }
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("Offline Muse OSC Stream Simulator")
        self.setGeometry(100, 100, 600, 400)
        self.setFixedSize(600, 400)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        # File section - EEG CSV
        eeg_file_layout = QHBoxLayout()
        self.load_eeg_button = QPushButton("Load EEG CSV")
        self.load_eeg_button.clicked.connect(self.load_eeg_file_dialog)
        eeg_file_layout.addWidget(self.load_eeg_button)
        
        self.eeg_file_label = QLabel("No EEG file loaded")
        eeg_file_layout.addWidget(self.eeg_file_label)
        eeg_file_layout.addStretch()
        
        layout.addLayout(eeg_file_layout)
        
        # File section - Sleep Stage CSV
        ss_file_layout = QHBoxLayout()
        self.load_ss_button = QPushButton("Load Sleep Stage CSV")
        self.load_ss_button.clicked.connect(self.load_ss_file_dialog)
        ss_file_layout.addWidget(self.load_ss_button)
        
        self.ss_file_label = QLabel("No sleep stage file loaded (optional)")
        ss_file_layout.addWidget(self.ss_file_label)
        ss_file_layout.addStretch()
        
        layout.addLayout(ss_file_layout)
        
        # Settings button
        settings_layout = QHBoxLayout()
        settings_layout.addStretch()
        self.settings_button = QPushButton("OSC Settings")
        self.settings_button.clicked.connect(self.show_osc_settings)
        settings_layout.addWidget(self.settings_button)
        
        layout.addLayout(settings_layout)
        
        # Status section
        self.status_label = QLabel("Ready")
        self.status_label.setAlignment(Qt.AlignCenter)
        font = QFont()
        font.setPointSize(12)
        self.status_label.setFont(font)
        layout.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.start_streaming)
        self.play_button.setEnabled(False)
        control_layout.addWidget(self.play_button)
        
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause_streaming)
        self.pause_button.setEnabled(False)
        control_layout.addWidget(self.pause_button)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_streaming)
        self.stop_button.setEnabled(False)
        control_layout.addWidget(self.stop_button)
        
        control_layout.addStretch()
        layout.addLayout(control_layout)
        
        # Speed control
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Playback Speed:"))
        
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(10, 300)  # 0.1x to 3x (multiplied by 100)
        self.speed_slider.setValue(100)  # 1.0x
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.setTickInterval(100)
        self.speed_slider.valueChanged.connect(self.update_speed)
        speed_layout.addWidget(self.speed_slider)
        
        self.speed_label = QLabel("1.0x")
        self.speed_label.setMinimumWidth(50)
        speed_layout.addWidget(self.speed_label)
        
        speed_layout.addStretch()
        layout.addLayout(speed_layout)
        
        # Info section
        info_text = (
            "This simulator streams CSV file data via OSC to simulate a Muse device.\n"
            f"OSC messages are sent to {self.target_ip}:{self.target_port}\n"
            "Ensure EazzZyLearn is running and listening on the same port."
        )
        info_label = QLabel(info_text)
        info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(info_label)
        
        layout.addStretch()
        
        # Apply some styling
        self.setStyleSheet("""
            QPushButton {
                min-width: 80px;
                min-height: 30px;
            }
            QPushButton:enabled {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #666666;
            }
            QPushButton:hover:enabled {
                background-color: #45a049;
            }
            QProgressBar {
                min-height: 20px;
                text-align: center;
            }
        """)
    
    def load_csv_files(self, eeg_filepath=None, ss_filepath=None):
        """Load data from CSV files."""
        try:
            # Use stored file paths if not provided
            if eeg_filepath:
                self.eeg_filepath = eeg_filepath
            if ss_filepath:
                self.ss_filepath = ss_filepath
                
            # Check if we have EEG data to load
            if not self.eeg_filepath:
                return False
            
            # Emit progress update if in thread
            if hasattr(self, 'loading_thread') and self.loading_thread and self.loading_thread.isRunning():
                self.loading_thread.progress_update.emit(f"Reading EEG file: {os.path.basename(self.eeg_filepath)}...")
                
            # Load EEG data
            eeg_df = pd.read_csv(self.eeg_filepath)
            
            # Extract timestamps and EEG channels
            self.timestamps = eeg_df['ts'].values
            self.timestamps = self.timestamps - self.timestamps[0]
            
            # Get EEG channels (ch1-ch6 or ch1-ch4)
            channel_cols = [col for col in eeg_df.columns if col.startswith('ch')]
            eeg_data = eeg_df[channel_cols].values
            
            # Pad to 6 channels (Muse standard)
            if eeg_data.shape[1] < 6:
                padding = np.zeros((eeg_data.shape[0], 6 - eeg_data.shape[1]))
                self.data = np.hstack([eeg_data, padding])
            else:
                self.data = eeg_data[:, :6]
            
            # Load sleep stage file if available
            if self.ss_filepath and os.path.exists(self.ss_filepath):
                # Emit progress update if in thread
                if hasattr(self, 'loading_thread') and self.loading_thread and self.loading_thread.isRunning():
                    self.loading_thread.progress_update.emit(f"Reading sleep stage file: {os.path.basename(self.ss_filepath)}...")
                    
                ss_df = pd.read_csv(self.ss_filepath)
                
                # Create sleep scores array matching EEG samples
                self.sleep_scores = np.zeros(len(self.data), dtype=int)
                
                # Emit progress update if in thread
                if hasattr(self, 'loading_thread') and self.loading_thread and self.loading_thread.isRunning():
                    self.loading_thread.progress_update.emit("Synchronizing sleep stages with EEG data...")
                
                # Interpolate sleep stages to match EEG timestamps
                if 'timestamps' in ss_df.columns:
                    ss_timestamps = ss_df['timestamps'].values
                    ss_timestamps = ss_timestamps - ss_timestamps[0]
                    sleep_stages = ss_df['sleep_stage'].values
                    
                    # Map sleep stages to indices
                    stage_indices = [self.sleep_stage_map.get(stage, 12) for stage in sleep_stages]
                    
                    # For each EEG timestamp, find the closest sleep stage
                    for i, eeg_ts in enumerate(self.timestamps):
                        # Find the closest timestamp in sleep stages
                        idx = np.argmin(np.abs(ss_timestamps - eeg_ts))
                        # Only use if within reasonable time window (e.g., 30 seconds)
                        if abs(ss_timestamps[idx] - eeg_ts) < 30:
                            self.sleep_scores[i] = stage_indices[idx]
                        else:
                            self.sleep_scores[i] = 12  # Default to Wake if too far
                else:
                    # If no timestamps, assume uniform distribution
                    sleep_stages = ss_df['sleep_stage'].values
                    stage_indices = [self.sleep_stage_map.get(stage, 12) for stage in sleep_stages]
                    
                    # Repeat each sleep stage for 30 seconds worth of samples
                    samples_per_stage = 30 * self.sample_rate  # 30 seconds at 256Hz
                    
                    for i, stage_idx in enumerate(stage_indices):
                        start_idx = i * samples_per_stage
                        end_idx = min((i + 1) * samples_per_stage, len(self.sleep_scores))
                        if start_idx < len(self.sleep_scores):
                            self.sleep_scores[start_idx:end_idx] = stage_idx
            else:
                self.sleep_scores = None
                if self.ss_filepath:
                    print(f"Sleep stage file not found: {self.ss_filepath}")
            
            # Reset playback position
            self.current_index = 0
            
            return True
            
        except Exception as e:
            print(f"Error loading CSV files: {e}")
            return False
    
    def connect_osc(self):
        """Initialize OSC client."""
        try:
            self.osc_client = udp_client.SimpleUDPClient(self.target_ip, self.target_port)
            return True
        except Exception as e:
            print(f"Error connecting OSC: {e}")
            return False
    
    def load_eeg_file_dialog(self):
        """Open file dialog to load EEG CSV file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select EEG CSV file",
            "",
            "CSV files (*_eeg.csv *.csv);;All files (*.*)"
        )
        
        if filepath:
            self.eeg_filepath = filepath
            self.eeg_file_label.setText(f"EEG: {os.path.basename(filepath)}")
            self.update_data_loading()
    
    def load_ss_file_dialog(self):
        """Open file dialog to load sleep stage CSV file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select Sleep Stage CSV file",
            "",
            "CSV files (*_ss.csv *.csv);;All files (*.*)"
        )
        
        if filepath:
            self.ss_filepath = filepath
            self.ss_file_label.setText(f"Sleep: {os.path.basename(filepath)}")
            self.update_data_loading()
    
    def update_data_loading(self):
        """Update data loading when files are selected."""
        if self.eeg_filepath:
            # Create progress dialog
            self.progress_dialog = QProgressDialog(
                "Loading CSV files...\n\nThis may take a while for large files.",
                None,  # No cancel button
                0, 0,  # Indeterminate progress
                self
            )
            self.progress_dialog.setWindowTitle("Loading Data")
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.progress_dialog.setMinimumDuration(0)  # Show immediately
            self.progress_dialog.show()
            
            # Process events to ensure dialog appears
            QApplication.processEvents()
            
            # Create and start loading thread
            self.loading_thread = FileLoadingThread(self, 'csv')
            self.loading_thread.loading_complete.connect(self.on_loading_complete)
            self.loading_thread.progress_update.connect(self.on_loading_progress)
            self.loading_thread.start()
    
    def on_loading_progress(self, message):
        """Update progress dialog with loading status."""
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.setLabelText(message)
            QApplication.processEvents()
    
    def on_loading_complete(self, success, message):
        """Handle completion of file loading."""
        # Close progress dialog
        if hasattr(self, 'progress_dialog'):
            self.progress_dialog.close()
            self.progress_dialog = None
        
        if success:
            self.update_status_after_load()
        else:
            QMessageBox.critical(self, "Error", f"Failed to load CSV files: {message}")
    
    def update_status_after_load(self):
        """Update status and UI after successful data loading."""
        if self.data is not None:
            # Update status
            num_samples = len(self.data)
            num_channels = self.data.shape[1]
            duration = num_samples / self.sample_rate
            
            status = f"Ready: {num_samples} samples, {num_channels} channels, {duration:.1f}s"
            if self.sleep_scores is not None:
                status += " (with sleep stages)"
            self.status_label.setText(status)
            
            # Reset progress
            self.progress_bar.setValue(0)
            self.current_index = 0
            
            # Enable play button
            self.play_button.setEnabled(True)
            
            QMessageBox.information(self, "Success", status)
    
    def start_streaming(self):
        """Start streaming in a separate thread."""
        if self.data is None:
            QMessageBox.critical(self, "Error", "No data loaded")
            return
        
        if not self.connect_osc():
            QMessageBox.critical(self, "Error", f"Failed to connect to OSC {self.target_ip}:{self.target_port}")
            return
        
        # Create and start streaming thread
        self.streaming_thread = StreamingThread(self)
        self.streaming_thread.progress_update.connect(self.update_progress)
        self.streaming_thread.stream_complete.connect(self.on_stream_complete)
        self.streaming_thread.start()
        
        # Update buttons
        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.stop_button.setEnabled(True)
        self.status_label.setText("Streaming...")
    
    def pause_streaming(self):
        """Pause streaming."""
        if self.streaming_thread:
            self.streaming_thread.stop()
            self.streaming_thread.wait()
            self.streaming_thread = None
        
        # Update buttons
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.status_label.setText("Paused")
    
    def stop_streaming(self):
        """Stop streaming and reset."""
        if self.streaming_thread:
            self.streaming_thread.stop()
            self.streaming_thread.wait()
            self.streaming_thread = None
        
        self.current_index = 0
        self.progress_bar.setValue(0)
        
        # Update buttons
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Stopped")
    
    def update_progress(self, progress, status):
        """Update progress bar and status."""
        self.progress_bar.setValue(progress)
        self.status_label.setText(status)
    
    def on_stream_complete(self):
        """Handle stream completion."""
        self.status_label.setText("Stream complete")
        self.play_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.streaming_thread = None
    
    def update_speed(self, value):
        """Update playback speed."""
        self.playback_speed = value / 100.0
        self.speed_label.setText(f"{self.playback_speed:.1f}x")
    
    def show_osc_settings(self):
        """Show OSC settings dialog."""
        dialog = OSCSettingsDialog(self, self.target_ip, self.target_port)
        if dialog.exec_():
            ip, port = dialog.get_settings()
            self.target_ip = ip
            try:
                self.target_port = int(port)
                # Update info label
                info_text = (
                    "This simulator streams CSV file data via OSC to simulate a Muse device.\n"
                    f"OSC messages are sent to {self.target_ip}:{self.target_port}\n"
                    "Ensure EazzZyLearn is running and listening on the same port."
                )
                # Find and update the info label
                for widget in self.centralWidget().findChildren(QLabel):
                    if "This simulator streams" in widget.text():
                        widget.setText(info_text)
                        break
            except ValueError:
                QMessageBox.critical(self, "Error", "Port must be a number")
    
    def closeEvent(self, event):
        """Handle window close event."""
        if self.streaming_thread:
            self.streaming_thread.stop()
            self.streaming_thread.wait()
        event.accept()


def main():
    """Main entry point."""
    app = QApplication(sys.argv)
    window = MuseOSCSimulator()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()