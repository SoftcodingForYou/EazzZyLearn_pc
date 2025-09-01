#!/usr/bin/env python3
"""
Debug script to capture ALL OSC messages on port 12345
"""

from pythonosc import dispatcher, osc_server
from threading import Thread
import time

def handle_any_message(address, *args):
    """Handle any incoming OSC message"""
    print(f"Received: {address} with {len(args)} args")
    if address == "/eeg" and args:
        print(f"  EEG data: {args[:4] if len(args) >= 4 else args}")
    elif args:
        print(f"  Args: {args[:5]}..." if len(args) > 5 else f"  Args: {args}")

def debug_osc():
    print("Starting OSC debug listener on 0.0.0.0:12345")
    print("This will capture messages from any IP address...")
    
    # Create dispatcher that captures ALL messages
    disp = dispatcher.Dispatcher()
    disp.set_default_handler(handle_any_message)
    
    # Listen on all interfaces (0.0.0.0) instead of just localhost
    server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", 12345), disp)
    server_thread = Thread(target=server.serve_forever)
    server_thread.daemon = True
    server_thread.start()
    
    print("Listening for OSC messages... Press Ctrl+C to stop")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping debug listener...")
        server.shutdown()

if __name__ == "__main__":
    debug_osc()