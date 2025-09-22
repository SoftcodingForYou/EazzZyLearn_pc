from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
                            QLabel, QLineEdit, QPushButton, QComboBox,
                            QRadioButton, QButtonGroup, QFileDialog,
                            QDialogButtonBox, QGroupBox, QSpinBox, QCheckBox,
                            QScrollArea, QWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QFont
import parameters as p
import os
from datetime import datetime


class SettingsDialog(QDialog):
    """
    Settings dialog for configuring session parameters dynamically.
    
    Allows users to configure:
    - Device type (Muse/OpenBCI)
    - Output directory
    - Subject information
    - Audio cue settings
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EazzZyLearn - Settings")
        self.setModal(True)
        self.setFixedSize(600, 700)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowMaximizeButtonHint)  # Remove maximize button
        
        # Set window icon (same as main window)
        icon_path = os.path.join(os.path.dirname(__file__), 'assets', 'icon.png')
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        
        # Store original parameters to allow cancellation
        self.original_params = self.get_current_parameters()
        
        self.init_ui()
        self.apply_stylesheet()
        self.load_current_settings()
        
    def init_ui(self):
        """Initialize the user interface."""
        # Main dialog layout
        dialog_layout = QVBoxLayout()
        dialog_layout.setContentsMargins(10, 10, 10, 10)

        # Title label (outside scroll area)
        title_label = QLabel("Session Configuration")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        dialog_layout.addWidget(title_label)

        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        # Create widget for scroll area content
        scroll_widget = QWidget()
        main_layout = QVBoxLayout(scroll_widget)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 25, 15)

        # Debugging options
        debug_group = QGroupBox("Debugging Options")
        debug_layout = QVBoxLayout()
        debug_layout.setContentsMargins(10, 10, 10, 10)

        self.offline_checkbox = QCheckBox("Offline mode (simulated EEG stream)")
        self.plot_checkbox = QCheckBox("Enable real-time signal plotting")
        self.sound_feedback_checkbox = QCheckBox("Sound-EEG feedback loop")

        debug_layout.addWidget(self.offline_checkbox)
        debug_layout.addWidget(self.plot_checkbox)
        debug_layout.addWidget(self.sound_feedback_checkbox)
        debug_group.setLayout(debug_layout)
        main_layout.addWidget(debug_group)
        
        # Device Selection Group
        device_group = QGroupBox("Device Selection")
        device_layout = QHBoxLayout()
        device_layout.setContentsMargins(10, 10, 10, 10)  # Added internal margins
        
        self.device_group = QButtonGroup()
        self.muse_radio = QRadioButton("Muse")
        self.openbci_radio = QRadioButton("OpenBCI")
        self.device_group.addButton(self.muse_radio)
        self.device_group.addButton(self.openbci_radio)
        
        device_layout.addWidget(self.muse_radio)
        device_layout.addWidget(self.openbci_radio)
        device_layout.addStretch()
        device_group.setLayout(device_layout)
        main_layout.addWidget(device_group)
        
        # Output Directory Group
        output_group = QGroupBox("Output Directory")
        output_layout = QHBoxLayout()
        output_layout.setContentsMargins(10, 10, 10, 10)  # Added internal margins
        
        self.output_dir_edit = QLineEdit()
        self.browse_button = QPushButton("Browse...")
        self.browse_button.clicked.connect(self.browse_output_dir)
        
        output_layout.addWidget(self.output_dir_edit)
        output_layout.addWidget(self.browse_button)
        output_group.setLayout(output_layout)
        main_layout.addWidget(output_group)
        
        # Subject Information Group
        subject_group = QGroupBox("Subject Information")
        subject_layout = QGridLayout()
        subject_layout.setContentsMargins(10, 10, 10, 10)  # Added internal margins
        subject_layout.setVerticalSpacing(10)  # Space between rows
        
        # Name
        subject_layout.addWidget(QLabel("Name:"), 0, 0)
        self.name_edit = QLineEdit()
        subject_layout.addWidget(self.name_edit, 0, 1)
        
        # Age
        subject_layout.addWidget(QLabel("Age:"), 1, 0)
        self.age_spin = QSpinBox()
        self.age_spin.setRange(0, 120)
        subject_layout.addWidget(self.age_spin, 1, 1)
        
        # Sex
        subject_layout.addWidget(QLabel("Sex:"), 2, 0)
        self.sex_combo = QComboBox()
        self.sex_combo.addItems(["Female", "Male"])
        subject_layout.addWidget(self.sex_combo, 2, 1)
        
        subject_group.setLayout(subject_layout)
        main_layout.addWidget(subject_group)
        
        # Audio Cue Settings Group
        audio_group = QGroupBox("Audio Cue Settings")
        audio_layout = QGridLayout()
        audio_layout.setContentsMargins(10, 10, 10, 10)  # Added internal margins
        audio_layout.setVerticalSpacing(10)  # Space between rows
        
        # Chosen cue
        audio_layout.addWidget(QLabel("Cue Sound:"), 0, 0)
        self.cue_combo = QComboBox()
        self.cue_combo.addItems(["gong", "heart"])
        self.cue_combo.setEditable(True)
        audio_layout.addWidget(self.cue_combo, 0, 1)
        
        # Background sound
        audio_layout.addWidget(QLabel("Background:"), 1, 0)
        self.background_edit = QLineEdit()
        audio_layout.addWidget(self.background_edit, 1, 1)
        
        # Cue interval
        audio_layout.addWidget(QLabel("Cue Interval (min):"), 2, 0)
        self.interval_edit = QLineEdit()
        audio_layout.addWidget(self.interval_edit, 2, 1)
        
        audio_group.setLayout(audio_layout)
        main_layout.addWidget(audio_group)
        
        # Use Current Date/Time button
        date_layout = QHBoxLayout()
        self.use_datetime_button = QPushButton("Use Current Date/Time for Output Directory")
        self.use_datetime_button.clicked.connect(self.use_current_datetime)
        self.use_datetime_button.setObjectName("dateTimeButton")
        date_layout.addWidget(self.use_datetime_button)
        date_layout.addStretch()
        main_layout.addLayout(date_layout)
        
        # Add some spacing before buttons
        main_layout.addSpacing(10)
        
        # Custom dialog buttons (matching main window style)
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.ok_button = QPushButton("OK")
        self.ok_button.setObjectName("primaryButton")
        self.ok_button.clicked.connect(self.accept)
        self.ok_button.setMinimumWidth(100)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setObjectName("secondaryButton")
        self.cancel_button.clicked.connect(self.reject)
        self.cancel_button.setMinimumWidth(100)
        
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)

        # Set scroll widget to scroll area
        scroll_area.setWidget(scroll_widget)

        # Add scroll area and buttons to dialog
        dialog_layout.addWidget(scroll_area)
        dialog_layout.addLayout(button_layout)

        self.setLayout(dialog_layout)
    
    def load_current_settings(self):
        """Load current settings from parameters module."""
        # Offline session
        self.offline_checkbox.setChecked(p.IS_OFFLINE_SESSION)

        # Debugging options
        self.plot_checkbox.setChecked(p.ENABLE_SIGNAL_PLOT)
        self.sound_feedback_checkbox.setChecked(p.SOUND_FEEDBACK_LOOP)

        # Device
        if p.DEVICE.get("Muse", False):
            self.muse_radio.setChecked(True)
        else:
            self.openbci_radio.setChecked(True)
        
        # Output directory
        self.output_dir_edit.setText(p.OUTPUT_DIR)
        
        # Subject info
        self.name_edit.setText(p.SUBJECT_INFO.get('name', 'Generic'))
        try:
            age = int(p.SUBJECT_INFO.get('age', '0'))
            self.age_spin.setValue(age)
        except ValueError:
            self.age_spin.setValue(0)
        
        sex = p.SUBJECT_INFO.get('sex', 'Female')
        index = self.sex_combo.findText(sex)
        if index >= 0:
            self.sex_combo.setCurrentIndex(index)
        
        # Audio settings
        self.cue_combo.setCurrentText(p.SUBJECT_INFO.get('chosencue', 'gong'))
        self.background_edit.setText(p.SUBJECT_INFO.get('background', ''))
        self.interval_edit.setText(p.SUBJECT_INFO.get('cueinterval', '1'))
    
    def browse_output_dir(self):
        """Open directory browser for output directory selection."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            self.output_dir_edit.text() or "./",
            QFileDialog.ShowDirsOnly
        )
        if directory:
            self.output_dir_edit.setText(directory)
    
    def use_current_datetime(self):
        """Set output directory to current date/time format."""
        now = datetime.now()
        date_str = now.strftime("%Y_%m_%d_%H%M")
        output_dir = f'./EazzZyLearn_output/{date_str}'
        self.output_dir_edit.setText(output_dir)
    
    def get_current_parameters(self):
        """Get current parameter values."""
        return {
            'device': dict(p.DEVICE),
            'output_dir': p.OUTPUT_DIR,
            'subject_info': dict(p.SUBJECT_INFO)
        }
    
    def apply_settings(self):
        """Apply the settings to the parameters module."""
        # Update offline session
        p.IS_OFFLINE_SESSION = self.offline_checkbox.isChecked()

        # Update debugging options
        p.ENABLE_SIGNAL_PLOT = self.plot_checkbox.isChecked()
        p.SOUND_FEEDBACK_LOOP = self.sound_feedback_checkbox.isChecked()

        # Update device
        p.DEVICE["Muse"] = self.muse_radio.isChecked()
        p.DEVICE["OpenBCI"] = self.openbci_radio.isChecked()
        
        # Update output directory
        p.OUTPUT_DIR = self.output_dir_edit.text()
        
        # Create output directory if it doesn't exist
        if not os.path.exists(p.OUTPUT_DIR):
            os.makedirs(p.OUTPUT_DIR, exist_ok=True)
        
        # Update subject info
        p.SUBJECT_INFO['name'] = self.name_edit.text()
        p.SUBJECT_INFO['age'] = str(self.age_spin.value())
        p.SUBJECT_INFO['sex'] = self.sex_combo.currentText()
        p.SUBJECT_INFO['chosencue'] = self.cue_combo.currentText()
        p.SUBJECT_INFO['background'] = self.background_edit.text()
        p.SUBJECT_INFO['cueinterval'] = self.interval_edit.text()
    
    def accept(self):
        """Handle OK button - apply settings and close."""
        self.apply_settings()
        super().accept()
    
    def reject(self):
        """Handle Cancel button - restore original settings and close."""
        # Restore original parameters
        p.DEVICE = self.original_params['device']
        p.OUTPUT_DIR = self.original_params['output_dir']
        p.SUBJECT_INFO = self.original_params['subject_info']
        super().reject()
    
    def apply_stylesheet(self):
        """Apply the same stylesheet as the main window."""
        self.setStyleSheet("""
            QDialog {
                background-color: #f5f5f5;
            }
            
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 20px;  /* More space above the group box */
                padding-top: 20px;  /* More space for content inside */
                padding-bottom: 10px;  /* Added bottom padding */
            }
            
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 2px 5px 2px 5px;  /* Added vertical padding to title */
                margin-top: 0px;  /* Raise title to sit on border */
                background-color: #f5f5f5;  /* Match dialog background */
                color: #333333;
            }
            
            QLabel {
                color: #333333;
            }
            
            QLineEdit, QComboBox, QSpinBox {
                padding: 8px 5px;  /* Increased vertical padding */
                min-height: 20px;  /* Minimum height to prevent cropping */
                border: 1px solid #cccccc;
                border-radius: 3px;
                background-color: white;
            }
            
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
                border: 2px solid #4CAF50;
            }
            
            QRadioButton {
                color: #333333;
                spacing: 5px;
            }
            
            QRadioButton::indicator {
                width: 15px;
                height: 15px;
            }
            
            QRadioButton::indicator:checked {
                background-color: #4CAF50;
                border: 2px solid #4CAF50;
                border-radius: 7px;
            }
            
            QRadioButton::indicator:unchecked {
                background-color: white;
                border: 2px solid #cccccc;
                border-radius: 7px;
            }
            
            QCheckBox {
                color: #333333;
                spacing: 5px;
            }
            
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
            }
            
            QCheckBox::indicator:checked {
                background-color: #4CAF50;
                border: 2px solid #4CAF50;
                border-radius: 3px;
            }
            
            QCheckBox::indicator:unchecked {
                background-color: white;
                border: 2px solid #cccccc;
                border-radius: 3px;
            }
            
            QPushButton {
                background-color: #f0f0f0;
                color: #333333;
                border: 1px solid #cccccc;
                padding: 10px 15px;  /* Increased padding */
                border-radius: 3px;
                font-weight: bold;
                min-height: 30px;  /* Increased minimum height */
            }
            
            QPushButton:hover {
                background-color: #e0e0e0;
                border: 1px solid #999999;
            }
            
            QPushButton:pressed {
                background-color: #d0d0d0;
            }
            
            QPushButton#primaryButton {
                background-color: #4CAF50;
                color: white;
                border: none;
            }
            
            QPushButton#primaryButton:hover {
                background-color: #45a049;
            }
            
            QPushButton#primaryButton:pressed {
                background-color: #3d8b40;
            }
            
            QPushButton#secondaryButton {
                background-color: #cccccc;
                color: #666666;
            }
            
            QPushButton#secondaryButton:hover {
                background-color: #bbbbbb;
            }
            
            QPushButton#dateTimeButton {
                background-color: #2196F3;
                color: white;
                border: none;
            }
            
            QPushButton#dateTimeButton:hover {
                background-color: #1976D2;
            }
        """)