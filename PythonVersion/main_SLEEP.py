from backend.backend import Backend
from frontend.frontend import Frontend
from PyQt5.QtWidgets import QApplication
import sys

def main():
    
    app = QApplication(sys.argv)
    processing_controller = Frontend()

    backend = Backend(processing_controller) # Initializes all methods and starts receiver

    processing_controller.show()
    app.exec_()

if __name__ == "__main__":
    main()