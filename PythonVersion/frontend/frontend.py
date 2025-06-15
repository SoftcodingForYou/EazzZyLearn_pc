from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QComboBox, QLabel
import sys
import parameters as p

class Frontend(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EazzZyLearn Control Interface")
        self.setGeometry(100, 100, 350, 100)

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
        
        # Create buttons
        self.start_button = QPushButton("Start")
        self.force_button = QPushButton("Force")
        self.stop_button = QPushButton("Pause")

        # Add widgets to layout
        layout.addWidget(channel_label)
        layout.addWidget(self.channel_combo)
        layout.addWidget(self.start_button)
        layout.addWidget(self.force_button)
        layout.addWidget(self.stop_button)

        # Connect button signals
        self.start_button.clicked.connect(self.start_stimulation)
        self.force_button.clicked.connect(self.force_stimulation)
        self.stop_button.clicked.connect(self.pause_stimulation)

        # Connect channel selection
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
        from PyQt5.QtWidgets import QMessageBox
        
        reply = QMessageBox.question(self, 'Confirm Exit',
                                   'Are you sure you want to exit?',
                                   QMessageBox.Yes | QMessageBox.No,
                                   QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            self.window_closed = True
            print("GUI stopped")
            event.accept()  # Accept the close event
        else:
            event.ignore()  # Ignore the close event

    def start_stimulation(self):
        print("Resume Stimulation")
        self.stimulation_state = 1
        self.start_button.setProperty("active", True)
        self.force_button.setProperty("active", False)
        self.stop_button.setProperty("active", False)
        self.force_style_update()

    def force_stimulation(self):
        print("Force stimulation")
        self.stimulation_state = -1
        self.start_button.setProperty("active", False)
        self.force_button.setProperty("active", True)
        self.stop_button.setProperty("active", False)
        self.force_style_update()

    def pause_stimulation(self):
        print("Paused stimulation")
        self.stimulation_state = 0
        self.start_button.setProperty("active", False)
        self.force_button.setProperty("active", False)
        self.stop_button.setProperty("active", True)
        self.force_style_update()

    def channel_changed(self, value):
        value = str(value)
        self.processing_channel = int(value[:value.find(':')])

def main():
    app = QApplication(sys.argv)
    window = Frontend()
    window.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 