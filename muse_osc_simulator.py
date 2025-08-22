#!/usr/bin/env python3
"""
Offline Muse OSC Stream Simulator
Loads offline EEG data from pickle files and streams it via OSC
to simulate a real Muse device for EazzZyLearn testing.
"""

import pickle
import numpy as np
import time
import sys
import os
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog,
                            QProgressBar, QSlider, QCheckBox, QDialog,
                            QLineEdit, QGridLayout, QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
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
                
                # Apply channel flipping if requested
                if self.simulator.flip_signal:
                    sample = sample * -1
                
                # Send EEG data
                self.simulator.osc_client.send_message("/eeg", sample.tolist())
                
                # Send sleep scores if available
                if self.simulator.sleep_scores is not None and self.simulator.current_index < len(self.simulator.sleep_scores):
                    sleep_stage = self.simulator.sleep_scores[self.simulator.current_index]
                    self.simulator.osc_client.send_message("/muse_metrics", [sleep_stage])
                
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
        self.flip_signal = True  # Default to flipped as per parameters
        
        self.streaming_thread = None
        
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
        
        # File section
        file_layout = QHBoxLayout()
        self.load_button = QPushButton("Load Pickle File")
        self.load_button.clicked.connect(self.load_file_dialog)
        file_layout.addWidget(self.load_button)
        
        self.file_label = QLabel("No file loaded")
        file_layout.addWidget(self.file_label)
        file_layout.addStretch()
        
        self.settings_button = QPushButton("OSC Settings")
        self.settings_button.clicked.connect(self.show_osc_settings)
        file_layout.addWidget(self.settings_button)
        
        layout.addLayout(file_layout)
        
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
        self.speed_slider.setRange(10, 1000)  # 0.1x to 10x (multiplied by 100)
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
        
        # Signal options
        options_layout = QHBoxLayout()
        self.flip_checkbox = QCheckBox("Flip Signal (*-1)")
        self.flip_checkbox.setChecked(True)  # Default to flipped
        self.flip_checkbox.stateChanged.connect(self.update_flip_signal)
        options_layout.addWidget(self.flip_checkbox)
        options_layout.addStretch()
        layout.addLayout(options_layout)
        
        # Info section
        info_text = (
            "This simulator streams pickle file data via OSC to simulate a Muse device.\n"
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
    
    def load_pickle_file(self, filepath):
        """Load data from pickle file."""
        try:
            with open(filepath, 'rb') as f:
                data_dict = pickle.load(f)
            
            # Validate required fields
            if 'data' not in data_dict:
                raise ValueError("Pickle file must contain 'data' field")
            
            self.data = np.array(data_dict['data'])
            
            # Handle optional fields
            if 'unix_ts' in data_dict:
                self.timestamps = np.array(data_dict['unix_ts'])
            else:
                # Generate timestamps based on sample rate
                num_samples = self.data.shape[0]
                self.timestamps = np.arange(num_samples) / self.sample_rate
            
            if 'sleep_scores' in data_dict:
                self.sleep_scores = np.array(data_dict['sleep_scores'])
            else:
                self.sleep_scores = None
            
            # Reset playback position
            self.current_index = 0
            
            # Ensure data is 2D (samples x channels)
            if len(self.data.shape) == 1:
                self.data = self.data.reshape(-1, 1)
            
            # Pad to 8 channels if necessary
            if self.data.shape[1] < 8:
                padding = np.zeros((self.data.shape[0], 8 - self.data.shape[1]))
                self.data = np.hstack([self.data, padding])
            elif self.data.shape[1] > 8:
                self.data = self.data[:, :8]
            
            return True
            
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            return False
    
    def connect_osc(self):
        """Initialize OSC client."""
        try:
            self.osc_client = udp_client.SimpleUDPClient(self.target_ip, self.target_port)
            return True
        except Exception as e:
            print(f"Error connecting OSC: {e}")
            return False
    
    def load_file_dialog(self):
        """Open file dialog to load pickle file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Select pickle file",
            "",
            "Pickle files (*.pkl);;All files (*.*)"
        )
        
        if filepath:
            if self.load_pickle_file(filepath):
                filename = os.path.basename(filepath)
                self.file_label.setText(f"Loaded: {filename}")
                
                # Update status
                num_samples = len(self.data)
                num_channels = self.data.shape[1]
                duration = num_samples / self.sample_rate
                
                status = f"Ready: {num_samples} samples, {num_channels} channels, {duration:.1f}s"
                self.status_label.setText(status)
                
                # Reset progress
                self.progress_bar.setValue(0)
                
                # Enable play button
                self.play_button.setEnabled(True)
                
                QMessageBox.information(self, "Success", f"Loaded {filename}\n{status}")
            else:
                QMessageBox.critical(self, "Error", "Failed to load file")
    
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
    
    def update_flip_signal(self, state):
        """Update signal flipping setting."""
        self.flip_signal = (state == Qt.Checked)
    
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
                    "This simulator streams pickle file data via OSC to simulate a Muse device.\n"
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