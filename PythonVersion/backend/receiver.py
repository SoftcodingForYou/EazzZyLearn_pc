from abc import abstractclassmethod
import parameters as p
import numpy as np
import socket
import json
import time
from threading import Thread
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

        # self.prep_socket(self.ip, self.port) # Connect to socket
        self.prep_osc_receiver(self.ip, self.port) # Connect to OSC receiver
        self.current_eeg_data = None
        
        # Initialize zeros buffer and time stamps
        self.buffer         = self.prep_buffer(self.num_channels, self.buffer_length)
        self.time_stamps    = self.prep_time_stamps(self.buffer_length)

        self.softstate      = 'enabled' # For forcing/pausing stimulations


    def prep_socket(self, ip, port):
        # Setup UDP protocol: connect to the UDP EEG streamer
        self.receiver_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.receiver_sock.bind((ip, port))


    def prep_osc_receiver(self, ip, port):
        # Setup OSC server to receive EEG data
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map("/eeg", self.handle_eeg_message)  # Map OSC address to handler
        
        # Create OSC server
        self.server = osc_server.ThreadingOSCUDPServer((ip, port), self.dispatcher)
        self.server_thread = Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()


    def handle_eeg_message(self, address, *args): # TODO: This does not get called when turning into Muse's OSC streamer
        # Handle incoming OSC message
        # args[0] should contain the EEG data
        self.current_eeg_data = np.array([args[0]]) * -1  # Keep the signal flipping logic

        
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

    def get_sample(self):
        # Get eeg samples from the OSC stream
        if self.current_eeg_data is not None:
            return self.current_eeg_data
        return np.zeros((1, self.num_channels))  # Return zeros if no data received yet

    def get_time_stamp(self):
        return round(time.perf_counter() * 1000 - self.start_time, 4)


    def fill_buffer(self):
        # This functions fills the buffer in self.buffer
        # that later can be accesed to perfom the real time analysis
        while True:
            # Get samples
            sample      = self.get_sample()
            time_stamp  = self.get_time_stamp()

            # Transpose vector
            sample = np.transpose(sample)

            # Concatenate vector
            update_buffer = np.concatenate((self.buffer, sample), axis=1)

            # save to new buffer
            self.buffer = update_buffer[:, 1:]

            # Save time_stamp
            self.time_stamps = np.append(self.time_stamps, time_stamp)
            self.time_stamps = self.time_stamps[1:]

            # real_time_algorithm
            self.real_time_algorithm(self.buffer, self.time_stamps)

            # Stop thread
            if self.stop:
                return


    @abstractclassmethod
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
        self.receiver_thread.start()


    def stop_receiver(self):
        # Change the status of self.stop to stop the recording
        self.stop = True
        time.sleep(2)  # Wait two seconds to stop de recording to be sure that everything stopped


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
        if key == "p" and self.softstate != 'paused':
            self.softstate = 'paused'
            line = str(timestamp) + ', Stimulation paused'
            stimhistory = open(outputfile, 'a') # Appending
            stimhistory.write(line + '\n')
            stimhistory.close()
            print('*** Stimulation paused! Press R to resume ...')
        elif key == "r" and self.softstate != 'enabled':
            self.softstate = 'enabled'
            line = str(timestamp) + ', Stimulation enabled'
            stimhistory = open(outputfile, 'a') # Appending
            stimhistory.write(line + '\n')
            stimhistory.close()
            print('Stimulation resumed')
        elif key == "f" and self.softstate != 'forced':
            self.softstate = 'forced'
            line = str(timestamp) + ', Stimulation forced, sleep/wake stages are estimated but ignored'
            stimhistory = open(outputfile, 'a') # Appending
            stimhistory.write(line + '\n')
            stimhistory.close()
            print('Stimulation forced, ignoring sleep/wake staging')
        elif key == "q":
            rsp = input('*** Are you sure you want to quit the program? Y/N: ')
            if rsp.lower() == 'yes' or rsp.lower() == 'y':
                line = str(timestamp) + ', Program quit irreversibly'
                stimhistory = open(outputfile, 'a') # Appending
                stimhistory.write(line + '\n')
                stimhistory.close()
                print('Program quit entirely')
                self.stop_receiver()
            else:
                line = str(timestamp) + ', Program almost exited. Resuming now...'
                stimhistory = open(outputfile, 'a') # Appending
                stimhistory.write(line + '\n')
                stimhistory.close()
                self.softstate = 'enabled'
                print('Program was NOT quit, resuming stimulation')


if __name__ == "__main__":
    import os; os.system('clear')
    receiver = Receiver()
    receiver.start_receiver()