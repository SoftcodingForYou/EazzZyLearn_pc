from abc import abstractmethod
import parameters as p
import numpy as np
import socket
import json
import time
from threading import Thread, Lock
from pythonosc import dispatcher, osc_server

class Receiver:

    def __init__(self):
        # =================================================================
        # Initialize receiver
        # -----------------------------------------------------------------
        # - Connect to socket
        # - Initialize zero buffer: self.buffer
        # - Initialize time stamps array of zeros: self.time_stamps
        # =================================================================

        # Set streaming parameters
        self.ip             = p.IP
        self.port           = p.PORT
        self.sample_rate    = p.SAMPLERATE

        # Set buffer parameters
        self.buffer_length  = p.MAIN_BUFFER_LENGTH
        self.num_channels   = p.NUM_CHANNELS

        # Stop recording
        self.stop           = False

        # Using Muse devices
        self.use_muse_sleep_classifier = p.USE_MUSE_SLEEP_CLASSIFIER
        self.current_muse_metrics = None

        # self.prep_socket(self.ip, self.port) # Connect to socket
        self.lock           = Lock()
        self.prep_osc_receiver(self.ip, self.port) # Connect to OSC receiver
        self.current_eeg_data = np.zeros((1, p.NUM_CHANNELS))
        
        # Initialize zeros buffer and time stamps
        self.buffer         = self.prep_buffer(self.num_channels, self.buffer_length)
        self.time_stamps    = self.prep_time_stamps(self.buffer_length)

        self.softstate      = 1

        self.received_new_sample = False
        

    def prep_socket(self, ip, port):
        # Setup UDP protocol: connect to the UDP EEG streamer
        self.receiver_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receiver_sock.bind((ip, port))


    def prep_osc_receiver(self, ip, port):
        # Setup OSC server to receive EEG data
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("/eeg", self.handle_eeg_message)  # Map OSC address to handler
        if self.use_muse_sleep_classifier:
            self.dispatcher.map("/muse_metrics", self.handle_muse_metrics_message)

        # Create OSC server
        self.server = osc_server.ThreadingOSCUDPServer((ip, port), self.dispatcher)
        self.server_thread = Thread(target=self.server.serve_forever, name='osc_server_thread')
        self.server_thread.daemon = True
        print(f"OSC server listening on {ip}:{port} for Muse OSC messages")


    def handle_eeg_message(self, address, *args):
        # Handle incoming OSC message from Muse headband
        # Muse sends:
        # - 4 EEG channels TP9, AF7, AF8, TP10
        # - 1 to 4 optional Aux channels which we do not consider
            
        with self.lock: # Queuing calls which would otherwise get executed in parallel
            if len(args) >= self.num_channels:
                # Take first 4 channels and pad with zeros to match NUM_CHANNELS
                muse_data = list(args[:self.num_channels]) 
                # Important! This drops NaN elements on the right edge of the args vector.
                # Therefore, we need to pad the vector to restitute the NaNs in the next step
                while len(muse_data) < self.num_channels:
                    muse_data.append(np.nan)

                new_eeg_data = np.array([muse_data])
                # Loop through each element and handle NaN values
                for i in range(new_eeg_data.shape[1]):
                    if np.isnan(new_eeg_data[0,i]):
                        new_eeg_data[0,i] = self.current_eeg_data[0,i]
                
                self.current_eeg_data = new_eeg_data
                self.received_new_sample = True

                self.execute_algorithm_pipeline()
            else:
                print(f"Warning: Received {len(args)} EEG values, expected at least 4")


    def handle_muse_metrics_message(self, address, *args):
        # Handle incoming OSC message from Muse headband
        # Muse sends 4 EEG channels: TP9, AF7, AF8, TP10
        muse_metrics = list(args)
        if self.current_muse_metrics is None:
            self.current_muse_metrics = np.zeros(len(muse_metrics))
        for i in range(len(muse_metrics)):
            self.current_muse_metrics[i] = muse_metrics[i]

        
    def prep_buffer(self, num_channels, length):
        # This functions creates the buffer structure
        # that will be filled with eeg datasamples
        return np.zeros((num_channels, length))


    def prep_time_stamps(self, length):
        return np.zeros(length)


    # def get_sample(self):
    #     # Get eeg samples from the UDP streamer
    #     raw_message, _  = self.receiver_sock.recvfrom(1024)
    #     eeg_data        = json.loads(raw_message)['data']  # Vector with length self.num_channel

    #     # Flip signals because OpenBCI streams data P - N pins which 
    #     # corresponds to Reference - Electrode
    #     eeg_data        = np.array([eeg_data]) * -1
        
    #     return eeg_data

    # def get_sample(self):
    #     # Get eeg samples from the OSC stream
    #     if self.current_eeg_data is not None:
    #         return self.current_eeg_data
    #     return np.zeros((1, self.num_channels))  # Return zeros if no data received yet

    def get_time_stamp(self):
        # return round(time.perf_counter() * 1000 - self.start_time, 4)
        return round(time.time_ns() / 1000000, 4)


    def fill_buffer(self):
        # This functions fills the buffer in self.buffer
        # that later can be accesed to perfom the real time analysis
        while True:

            if not self.received_new_sample:
                if hasattr(self, 'gui') and self.gui.window_closed:
                    self.HndlDt.disk_io.close_files()
                    self.Stg.disk_io.close_files()
                    self.Pdct.disk_io.close_files()
                    self.Cng.disk_io.close_files()
                    self.define_stimulation_state(2, self.HndlDt.stim_path, self.get_time_stamp())
                    return
                
                time.sleep(0.001)  # Add a small delay to reduce CPU usage
                continue
            else:
                self.received_new_sample = False

            # Get samples
            sample      = self.current_eeg_data # self.get_sample()
            time_stamp  = self.get_time_stamp()

            # Transpose vector
            sample = np.transpose(sample)

            # save to new buffer
            self.buffer = np.concatenate((self.buffer[:, 1:], sample), axis=1)

            # Save time_stamp
            self.time_stamps[:-1] = self.time_stamps[1:]
            self.time_stamps[-1] = time_stamp

            # real_time_algorithm
            self.real_time_algorithm(self.buffer, self.time_stamps)

            # Stop thread
            if self.stop:
                return
            

    def execute_algorithm_pipeline(self):
        sample      = self.current_eeg_data
        time_stamp  = self.get_time_stamp()

        # Transpose vector
        sample = np.transpose(sample)

        # save to new buffer
        self.buffer = np.concatenate((self.buffer[:, 1:], sample), axis=1)

        # Save time_stamp
        self.time_stamps[:-1] = self.time_stamps[1:]
        self.time_stamps[-1] = time_stamp

        # real_time_algorithm
        self.real_time_algorithm(self.buffer, self.time_stamps)


    @abstractmethod
    def real_time_algorithm(self, buffer, time_stamps):
        # This method is overwritten on class Backend
        pass


    def start_receiver(self, output_dir, subject_info):
        # Define thread for receiving
        self.receiver_thread = Thread(
            target=self.fill_buffer,
            name='receiver_thread',
            daemon=False)
        # Set start time
        self.start_time = time.perf_counter() * 1000  # Get recording start time
        # start thread
        # self.receiver_thread.start()
        self.server_thread.start()


    def stop_receiver(self):
        # Change the status of self.stop to stop the recording
        self.stop = True
        self.server.shutdown()
        line = str(self.get_time_stamp()) + ', Program quit irreversibly'
        self.Cng.disk_io.line_store(line, self.HndlDt.stim_path)
        self.HndlDt.disk_io.close_files()
        self.Stg.disk_io.close_files()
        self.Pdct.disk_io.close_files()
        self.Cng.disk_io.close_files()
        time.sleep(2)  # Wait two seconds to stop de recording to be sure that everything stopped
        print('Program quit entirely')


    def define_stimulation_state(self, key, outputfile, timestamp):
        # =================================================================
        # This method has the objective to handle the stimulations manually
        # at will. It will write the information about pausing,
        # esuming, ... into outputfile.
        # R  =  Stimulation enabled and called when detecting Slow 
        #       Oscillation downstate (default state)
        # P  =  Stimulation paused, preventing calling stimulation when 
        #       detecting Slow Oscillations
        # F  =  Forcing stimulations when Slow Oscillations are detected,
        #       ignoring the sleep/wake stage predictions
        # Q  =  Quitting program irreversibly (checkpoint before quitting) 
        # =================================================================
        if key == 0:
            self.softstate = key
            line = str(self.get_time_stamp()) + ', Stimulation paused'
            self.Cng.disk_io.line_store(line, self.HndlDt.stim_path)
            print('*** Stimulation paused! ...')
        elif key == 1:
            self.softstate = key
            line = str(self.get_time_stamp()) + ', Stimulation enabled'
            self.Cng.disk_io.line_store(line, self.HndlDt.stim_path)
            print('Stimulation resumed')
        elif key == -1:
            self.softstate = key
            line = str(self.get_time_stamp()) + ', Stimulation forced, sleep/wake stages are estimated but ignored'
            self.Cng.disk_io.line_store(line, self.HndlDt.stim_path)
            print('Stimulation forced, ignoring sleep/wake staging')