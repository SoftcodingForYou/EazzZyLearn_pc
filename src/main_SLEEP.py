from backend.backend import Backend
from frontend.frontend import Frontend
from frontend.settings_dialog import SettingsDialog
from PyQt5.QtWidgets import QApplication, QMessageBox
import sys

def main():
    
    app = QApplication(sys.argv)
    
    # Show settings dialog before initializing
    settings_dialog = SettingsDialog()
    result = settings_dialog.exec_()
    
    # If user cancelled, exit
    if result != SettingsDialog.Accepted:
        QMessageBox.information(None, "Setup Cancelled", 
                               "Session setup was cancelled. Exiting...")
        sys.exit(0)
    
    # Now initialize with the configured parameters
    processing_controller = Frontend()

    backend = Backend(processing_controller) # Initializes all methods and starts receiver
    
    # Store backend reference for cleanup
    processing_controller.backend = backend

    processing_controller.show()
    app.exec_()

if __name__ == "__main__":
    main()