#!/usr/bin/env python3
"""
Simple test script to verify OSC connection with Muse headband
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'PythonVersion'))

from PythonVersion.backend.receiver import Receiver
import time

def test_osc_connection():
    print("Starting OSC connection test...")
    print("Make sure your Muse headband is streaming to port 12345")
    print("Now listening on 0.0.0.0:12345 (all interfaces)")
    
    try:
        # Create receiver instance
        receiver = Receiver()
        
        print("OSC server started. Waiting for EEG data...")
        print("Press Ctrl+C to stop")
        
        # Let it run for a while to receive data
        start_time = time.time()
        while True:
            time.sleep(1)
            
            # Check if we're receiving data
            if receiver.current_eeg_data is not None:
                elapsed = time.time() - start_time
                print(f"[{elapsed:.1f}s] Current EEG data shape: {receiver.current_eeg_data.shape}")
                print(f"[{elapsed:.1f}s] Sample values: {receiver.current_eeg_data[0][:4]}")  # Show first 4 channels
                print(f"[{elapsed:.1f}s] Total messages received: {receiver.message_count}")
            else:
                elapsed = time.time() - start_time
                print(f"[{elapsed:.1f}s] No EEG data received yet...")
                
    except KeyboardInterrupt:
        print("\nStopping test...")
        receiver.stop_receiver()
        print("Test completed.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_osc_connection()