from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QPushButton, QComboBox, QLabel, QCheckBox,
                            QMenuBar, QMenu, QAction, QMessageBox)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from frontend.pyqt_native_plot_widget import NativePlotWidget
import sys
import parameters as p
import os
import time

class Frontend(QMainWindow):
    """
    Main GUI window for EazzZyLearn.

    Features:
    - Channel selection dropdown for processing channel.
    - Enable, Force, and Pause buttons to control stimulation state.
    - Status and speed labels for real-time feedback.
    - Custom window icon and styling.
    - Handles window close events with confirmation dialog.
    - Methods to update UI elements and respond to user actions.

    Usage:
        window = Frontend()
        window.show()
    """
    
    def __init__(self):
        """Initialize the main GUI window for EazzZyLearn.

        Sets up the window title, icon, size, and disables the maximize button.
        Creates and arranges all UI elements, including:
            - Channel selection dropdown
            - Enable, Force, and Pause buttons
            - Status and speed labels
        Connects button and dropdown signals to their respective handlers.
        Initializes default states for channel selection and stimulation controls.
        Applies custom styles to the buttons and window.
        """

        super().__init__()
        self.setWindowTitle("EazzZyLearn")
        self.setGeometry(100, 100, 700, 500)  # Larger window for plot
        # self.setFixedSize(350, 200)  # Lock window size
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowMaximizeButtonHint)  # Remove maximize button

        # Set window icon
        icon_path = os.path.join(os.path.dirname(__file__), 'assets', 'icon.png')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # Create menu bar
        self.create_menu_bar()

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create channel selection dropdown
        channel_label = QLabel("Current processing channel:")
        option_list = [f"{value+1}: {key}" for key, value in p.ELEC.items()]
        self.channel_combo = QComboBox()
        for i in range(0, p.NUM_CHANNELS):
            self.channel_combo.addItem(option_list[i])

        # Create flip signal checkbox
        self.flip_signal = p.FLIP_SIGNAL
        self.flip_signal_checkbox = QCheckBox("Flip signal")
        
        # Create buttons
        self.start_button = QPushButton("Enable")
        self.force_button = QPushButton("Force")
        self.stop_button = QPushButton("Pause")
        
        # Create status label
        self.status_label = QLabel("Initializing ...")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #333333;
                font-size: 12px;
                padding: 5px;
            }
        """)

        self.speed_label = QLabel("Initializing ...")
        self.speed_label.setStyleSheet("""
            QLabel {
                color: #333333;
                font-size: 12px;
                padding: 5px;
            }
        """)

        self.stage_label = QLabel("Unknown stage ...")
        self.stage_label.setStyleSheet("""
            QLabel {
                color: #333333;
                font-size: 12px;
                padding: 5px;
            }
        """)

        # Create native PyQt5 plot widget
        self.plot_widget = NativePlotWidget(p.MAIN_BUFFER_LENGTH, p.SAMPLERATE)

        # Add widgets to layout
        layout.addWidget(channel_label)
        layout.addWidget(self.channel_combo)
        layout.addWidget(self.flip_signal_checkbox)
        layout.addWidget(self.start_button)
        layout.addWidget(self.force_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.status_label)
        layout.addWidget(self.speed_label)
        layout.addWidget(self.stage_label)
        layout.addWidget(self.plot_widget)

        # Connect button signals
        self.start_button.clicked.connect(self.start_stimulation)
        self.force_button.clicked.connect(self.force_stimulation)
        self.stop_button.clicked.connect(self.pause_stimulation)

        # Connect channel selection
        self.flip_signal_checkbox.stateChanged.connect(self.flip_signal_changed)
        self.channel_combo.currentTextChanged.connect(self.channel_changed)

        self.window_closed = False
        self.processing_channel = p.IDX_ELEC
        self.stimulation_state = 1 # 1 Started; 0 Paused, -1 Forced

        # Defaults
        self.channel_combo.setCurrentText(option_list[p.IDX_ELEC])
        self.start_button.setProperty("active", True)
        self.force_button.setProperty("active", False)
        self.stop_button.setProperty("active", False)
        self.set_stylesheet()
    
    def create_menu_bar(self):
        """Create the menu bar with File and Settings menus."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        # Settings action
        settings_action = QAction('Session Settings...', self)
        settings_action.setShortcut('Ctrl+S')
        settings_action.triggered.connect(self.show_settings_dialog)
        file_menu.addAction(settings_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        # About action
        about_action = QAction('About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def show_settings_dialog(self):
        """Show the settings dialog for runtime configuration."""
        from frontend.settings_dialog import SettingsDialog
        
        # Warn user if recording is in progress
        if hasattr(self, 'backend') and hasattr(self.backend, 'monitor_running'):
            if self.backend.monitor_running:
                reply = QMessageBox.warning(self, 'Recording in Progress',
                                          'A recording session is currently active. '
                                          'Changing settings now will only affect the next session. '
                                          'Continue?',
                                          QMessageBox.Yes | QMessageBox.No,
                                          QMessageBox.No)
                if reply != QMessageBox.Yes:
                    return
        
        dialog = SettingsDialog(self)
        if dialog.exec_() == SettingsDialog.Accepted:
            QMessageBox.information(self, 'Settings Updated',
                                  'Settings have been updated. '
                                  'They will take effect in the next session.')
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(self, 'About EazzZyLearn',
                        'EazzZyLearn v2025.06\n\n'
                        'Real-time closed-loop neurofeedback system\n'
                        'for sleep research and memory consolidation.\n\n'
                        'Powered by Muse EEG technology.')

    def set_stylesheet(self):
        """Set the stylesheet for the buttons"""
        self.setStyleSheet("""
            QPushButton[active="true"] {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton[active="false"] {
                background-color: #cccccc;
                color: #666666;
                border: none;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton[active="false"]:hover {
                background-color: #bbbbbb;
            }
        """)

    def force_style_update(self):
        self.start_button.style().unpolish(self.start_button)
        self.start_button.style().polish(self.start_button)
        self.force_button.style().unpolish(self.force_button)
        self.force_button.style().polish(self.force_button)
        self.stop_button.style().unpolish(self.stop_button)
        self.stop_button.style().polish(self.stop_button)

    def closeEvent(self, event):
        """Handle window close event"""
        from PyQt5.QtWidgets import QMessageBox, QApplication
        
        reply = QMessageBox.question(self, 'Confirm Exit',
                                   'Are you sure you want to exit?',
                                   QMessageBox.Yes | QMessageBox.No,
                                   QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.window_closed = True
            # Give backend time to detect window_closed flag
            time.sleep(0.5)
            # Force stop if backend exists and hasn't stopped yet
            if hasattr(self, 'backend') and self.backend and not self.backend.stop:
                print("Forcing backend shutdown...")
                self.backend.monitor_running = False
                self.backend.stop_receiver()
                time.sleep(0.5)  # Give time for threads to cleanup
            print("GUI stopped")
            event.accept()  # Accept the close event
            # Ensure application terminates
            QApplication.quit()
        else:
            event.ignore()  # Ignore the close event

    def start_stimulation(self):
        if self.stimulation_state == 1:
            return
        print("Enabling Stimulation")
        self.stimulation_state = 1
        self.start_button.setProperty("active", True)
        self.force_button.setProperty("active", False)
        self.stop_button.setProperty("active", False)
        self.force_style_update()

    def force_stimulation(self):
        if self.stimulation_state == -1:
            return
        print("Forcing stimulation")
        self.stimulation_state = -1
        self.start_button.setProperty("active", False)
        self.force_button.setProperty("active", True)
        self.stop_button.setProperty("active", False)
        self.force_style_update()

    def pause_stimulation(self):
        if self.stimulation_state == 0:
            return
        print("Pausing stimulation")
        self.stimulation_state = 0
        self.start_button.setProperty("active", False)
        self.force_button.setProperty("active", False)
        self.stop_button.setProperty("active", True)
        self.force_style_update()

    def flip_signal_changed(self, state):
        """Handle flip signal checkbox state change"""
        self.flip_signal = state == Qt.Checked
        # Notify backend of state change
        if hasattr(self, 'backend'):
            self.backend.set_flip_signal(self.flip_signal)

    def channel_changed(self, value):
        value = str(value)
        self.processing_channel = int(value[:value.find(':')])

    def update_status_text(self, text):
        """Update the status label text"""
        if not self.window_closed and self.status_label:
            self.status_label.setText(f"{text}")

    def update_speed_text(self, text):
        """Update the speed label text"""
        if not self.window_closed and self.speed_label:
            self.speed_label.setText(f"{text}")
    
    def update_sleep_states(self, is_awake, is_sws):
        """Update the sleep state display"""
        if not self.window_closed and self.status_label:
            wake_text = f"Awake: {is_awake}"
            deep_sleep_text = f"Deep Sleep: {is_sws}"
            self.stage_label.setText(f"{wake_text} | {deep_sleep_text}")
    
    def update_plot(self, buffer_data):
        """Update the EEG plot with new buffer data"""
        if not self.window_closed and buffer_data is not None:
            try:
                # Update plot data using native widget
                self.plot_widget.update_data(buffer_data)
            except Exception as e:
                print(f"Plot update error: {e}")

def main():
    app = QApplication(sys.argv)
    window = Frontend()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 