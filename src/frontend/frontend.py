from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QComboBox, QLabel, 
                            QCheckBox, QAction, QMessageBox, QGroupBox)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, pyqtSignal, pyqtSlot, QMetaObject, Q_ARG
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

        # Create flip signal checkbox
        flip_group = QGroupBox("Signal polarity")
        flip_layout = QVBoxLayout()
        flip_layout.setContentsMargins(10, 10, 10, 10)
        self.flip_signal = p.FLIP_SIGNAL
        self.flip_signal_checkbox = QCheckBox("Flip signal")
        flip_layout.addWidget(self.flip_signal_checkbox)
        flip_group.setLayout(flip_layout)
        flip_group.setToolTip("This multiplies the incoming signal by -1 and will affect both online processing and offline stored signals.")

        # Signal processing
        processing_group = QGroupBox("Online processing")
        processing_layout = QVBoxLayout()
        # List channels for dropdown menus
        option_list = [f"{value+1}: {key}" for key, value in p.ELEC.items()]

        #       Create channel selection dropdown
        channel_label = QLabel("Real-time channel:")
        self.channel_combo = QComboBox()
        for i in range(0, p.NUM_CHANNELS):
            self.channel_combo.addItem(option_list[i])

        #       Create online reference selection dropdown
        reference_label = QLabel("Online reference channel:")
        self.online_ref_combo = QComboBox()
        self.online_ref_combo.addItem("0: None")
        for i in range(0, p.NUM_CHANNELS):
            self.online_ref_combo.addItem(option_list[i])

        processing_channel_group = QWidget()
        processing_channel_layout = QHBoxLayout()
        processing_channel_layout.addWidget(channel_label)
        processing_channel_layout.addWidget(self.channel_combo)
        processing_channel_group.setLayout(processing_channel_layout)
        reference_channel_group = QWidget()
        reference_channel_layout = QHBoxLayout()
        reference_channel_layout.addWidget(reference_label)
        reference_channel_layout.addWidget(self.online_ref_combo)
        reference_channel_group.setLayout(reference_channel_layout)
        processing_layout.addWidget(processing_channel_group)
        processing_layout.addWidget(reference_channel_group)
        processing_group.setLayout(processing_layout)
        processing_group.setToolTip("Select channels for real-time downstate stimulation and for online re-referencing the real-time channel.\nThis is only affecting online processing and will not alter stored signals!")

        # Create stimulation state group
        stimulation_group = QGroupBox("Stimulation state")
        stimulation_layout = QHBoxLayout()
        self.start_button = QPushButton("Enable")
        self.force_button = QPushButton("Force")
        self.stop_button = QPushButton("Pause")
        stimulation_layout.addWidget(self.start_button)
        stimulation_layout.addWidget(self.force_button)
        stimulation_layout.addWidget(self.stop_button)
        stimulation_group.setLayout(stimulation_layout)
        stimulation_group.setToolTip("Set stimulation state:\nEnable: Stimulation will be applied during detected downstates in adequate sleep stages.\nForce: Stimulation will be applied at every detected downstate, regardless of sleep stage.\nPause: No stimulation will be applied.")

        # Initialize debugging settings from parameters
        self.sound_feedback_loop_enabled = p.SOUND_FEEDBACK_LOOP # Used in real_time_algorithm()
        self.plot_enabled = p.ENABLE_SIGNAL_PLOT
        
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

        # Create native PyQt5 plot widget (only if enabled)
        if self.plot_enabled:
            self.plot_widget = NativePlotWidget(p.MAIN_BUFFER_LENGTH, p.SAMPLERATE)
        else:
            self.plot_widget = None

        # Add widgets to layout
        layout.addWidget(flip_group)
        layout.addWidget(processing_group)
        layout.addWidget(stimulation_group)
        layout.addWidget(self.status_label)
        layout.addWidget(self.speed_label)
        layout.addWidget(self.stage_label)
        if self.plot_widget:
            layout.addWidget(self.plot_widget)

        # Connect button signals
        self.start_button.clicked.connect(self.start_stimulation)
        self.force_button.clicked.connect(self.force_stimulation)
        self.stop_button.clicked.connect(self.pause_stimulation)

        # Connect channel selection
        self.flip_signal_checkbox.stateChanged.connect(self.flip_signal_changed)
        self.channel_combo.currentTextChanged.connect(self.channel_changed)

        # Connect online reference selection
        self.online_ref_combo.currentTextChanged.connect(self.online_reference_changed)

        self.window_closed = False
        self.processing_channel = p.IDX_ELEC
        self.reference_channel = p.IDX_REF
        self.stimulation_state = 1 # 1 Started; 0 Paused, -1 Forced

        # Defaults
        self.channel_combo.setCurrentText(option_list[p.IDX_ELEC])
        self.flip_signal_checkbox.setChecked(p.FLIP_SIGNAL)
        self.online_ref_combo.setCurrentText(option_list[p.IDX_REF] if p.IDX_REF != -1 else "0: None")
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
                        'EazzZyLearn v2025.11\n\n'
                        'Real-time closed-loop neurofeedback system\n'
                        'for sleep research and memory consolidation.')

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

        if hasattr(self, 'backend') and hasattr(self.backend, 'SgPrc') and hasattr(self.backend, 'HndlDt'):
            # We take care of this here because the real time loop might not be running yet and we would miss channel switches
            self.backend.SgPrc.switch_channel(
                self.processing_channel, self.backend.HndlDt.stim_path, self.backend.current_time)
            
    def online_reference_changed(self, value):
        value = str(value)
        self.reference_channel = int(value[:value.find(':')])

        if hasattr(self, 'backend') and hasattr(self.backend, 'SgPrc') and hasattr(self.backend, 'HndlDt'):
            # We take care of this here because the real time loop might not be running yet and we would miss channel switches
            self.backend.SgPrc.switch_online_reference_channel(
                self.reference_channel, self.backend.HndlDt.stim_path, self.backend.current_time)

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
    
    # Thread-safe signal for plot updates
    plot_update_signal = pyqtSignal(object, object)
    
    def update_plot(self, buffer_data, buffer_data2=None):
        """Thread-safe update of the EEG plot with new buffer data"""
        if not self.window_closed and buffer_data is not None:
            try:
                # Make copies to avoid threading issues
                buffer_copy = buffer_data.copy() if buffer_data is not None else None
                buffer2_copy = buffer_data2.copy() if buffer_data2 is not None else None
                
                # Use Qt's thread-safe method invocation
                QMetaObject.invokeMethod(self, "_update_plot_gui",
                                        Qt.QueuedConnection,
                                        Q_ARG(object, buffer_copy),
                                        Q_ARG(object, buffer2_copy))
            except Exception as e:
                print(f"Plot update error: {e}")
    
    @pyqtSlot(object, object)
    def _update_plot_gui(self, buffer_data, buffer_data2):
        """GUI thread update of plot widget"""
        if not self.window_closed:
            try:
                self.plot_widget.update_data(buffer_data, buffer_data2)
            except Exception as e:
                print(f"Plot widget update error: {e}")

def main():
    app = QApplication(sys.argv)
    window = Frontend()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 