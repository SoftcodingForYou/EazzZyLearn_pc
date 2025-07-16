from abc import abstractmethod
import parameters as p
import numpy as np
import socket
import json
import time
from threading import Thread
import queue
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

        # Device parameters
        if len([bool(v) for v in p.DEVICE.values() if v]) != 1:
            raise Exception("Receiver::__init__() Use only one device at a time")

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

        if p.DEVICE["OpenBCI"]:
            self.prep_udp_server(self.ip, self.port) # Connect to net
            self.get_current_sample_and_timestamp = self.get_current_udp_sample
        elif p.DEVICE["Muse"]:
            self.eeg_queue = queue.SimpleQueue()
            self.prep_osc_server(self.ip, self.port) # Connect to OSC
            self.get_current_sample_and_timestamp = self.get_current_osc_sample
        self.current_eeg_data = np.zeros((1, p.NUM_CHANNELS))

        # Define thread for receiving
        self.receiver_thread = Thread(
            target=self.fill_buffer,
            name='receiver_thread',
            daemon=False)
        
        # Initialize zeros buffer and time stamps
        self.buffer         = self.prep_buffer(self.num_channels, self.buffer_length)
        self.time_stamps    = self.prep_time_stamps(self.buffer_length)

        self.softstate      = 1
        self.flip_signal    = p.FLIP_SIGNAL
        self._process_sample = self._process_sample_flipped if p.FLIP_SIGNAL else self._process_sample_normal
        

    def prep_udp_server(self, ip, port):
        """Sets up a UDP socket for receiving EEG data streams.

        This method initializes a UDP socket and binds it to the specified IP address and port.
        It is typically used to connect to an EEG data streamer (e.g., OpenBCI) that sends data
        over UDP. The socket is stored as `self.receiver_sock` for later use in receiving data.

        Args:
            ip (str): The IP address to bind the UDP socket to.
            port (int): The port number to bind the UDP socket to.

        """

        self.receiver_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receiver_sock.bind((ip, port))


    def prep_osc_server(self, ip, port):
        """Sets up an OSC (Open Sound Control) server to receive EEG and optional Muse metrics data.

        This method initializes an OSC server using the specified IP address and port.
        It maps incoming OSC messages with the address "/eeg" to the `handle_eeg_message` method,
        which processes EEG data from the Muse headband. If Muse sleep classification is enabled,
        it also maps "/muse_metrics" messages to the `handle_muse_metrics_message` method.

        The OSC server runs in a separate daemon thread to handle incoming messages asynchronously.

        Args:
            ip (str): The IP address to bind the OSC server to.
            port (int): The port number to bind the OSC server to.

        Side Effects:
            - Initializes and starts an OSC server in a background thread.
            - Sets up message handlers for EEG and Muse metrics data.
            - Prints a message indicating the server is listening.
        """
        
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
            self.eeg_queue.put((new_eeg_data, self.get_time_stamp()))
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


    def get_current_udp_sample(self):
        """Receives a single EEG data sample from the UDP streamer in blocking mode.
        This function is suitable for OpenBCI which can forward data over UDP protocols.

        This method waits (blocks) until a UDP packet is received on the configured socket.
        The received message is expected to be a JSON-encoded dictionary containing a 'data' field,
        which holds a list of EEG channel values. The function parses the message, extracts the EEG data,
        and flips the signal polarity (multiplies by -1) to match the expected reference orientation.

        Returns:
            np.ndarray: A 2D NumPy array containing the EEG data sample with flipped polarity.
        """

        raw_message, _  = self.receiver_sock.recvfrom(1024)
        eeg_data        = json.loads(raw_message)['data']  # Vector with length self.num_channel

        # Flip signals because OpenBCI streams data P - N pins which 
        # corresponds to Reference - Electrode
        eeg_data        = np.array([eeg_data])
        
        return eeg_data, self.get_time_stamp()


    def get_time_stamp(self):
        # return round(time.perf_counter() * 1000 - self.start_time, 4)
        return round(time.time_ns() / 1000000, 4)
    

    def get_current_osc_sample(self):
        return self.eeg_queue.get(block=True) # Wait for message if none is in queue


    def fill_buffer(self):
        # This functions fills the buffer in self.buffer
        # that later can be accesed to perfom the real time analysis
        while True:

            # Get samples
            sample, time_stamp = self.get_current_sample_and_timestamp()
            sample = self._process_sample(sample)

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
            

    def set_flip_signal(self, value):
        if self.flip_signal != value:
            self.flip_signal = value
            # Set function pointer based on state
            self._process_sample = self._process_sample_flipped if value else self._process_sample_normal
            print(f"Flip signal: {'Enabled' if value else 'Disabled'}")


    def _process_sample_normal(self, sample):
        return sample


    def _process_sample_flipped(self, sample):
        return sample * -1


    @abstractmethod
    def real_time_algorithm(self, buffer, time_stamps):
        # This method is overwritten on class Backend
        pass


    def start_receiver(self, output_dir, subject_info):
        # Set start time
        # start threads
        self.receiver_thread.start()
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


    def define_stimulation_state(self, key):
        """Handles manual control of the stimulation state and logs the action to the output file.

        This method allows manual switching between different stimulation states:
            1 (R): Stimulation enabled (default) — triggers stimulation when a Slow Oscillation downstate is detected.
            0 (P): Stimulation paused — prevents stimulation even if a Slow Oscillation is detected.
           -1 (F): Stimulation forced — always triggers stimulation on Slow Oscillation detection, ignoring sleep/wake stage predictions.
            Q:     Quits the program irreversibly (checkpoint before quitting).

        The method writes a log entry describing the state change to the output file.

        Args:
            key (int): The stimulation state to set (1=enabled, 0=paused, -1=forced).
            outputfile: The file to which the state change should be logged.
            timestamp: The current timestamp for the log entry.
        """
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