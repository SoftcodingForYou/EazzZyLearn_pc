from abc import abstractclassmethod
import parameters as p
import numpy as np
import serial #Crucial: Install using pip3 install "pyserial", NOT "serial"
import serial.tools.list_ports
import keyboard
import json
import time
from threading import Thread

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
        self.sample_rate    = p.SAMPLERATE
        self.pga            = p.PGA
        self.baud_rate      = p.BAUDRATE
        self.time_out       = p.TIMEOUT
        self.search_device()
        self.connect_board()

        # Set buffer parameters
        self.buffer_length  = p.MAIN_BUFFER_LENGTH
        self.num_channels   = p.NUM_CHANNELS

        # Stop recording
        self.stop           = False
        
        # Initialize zeros buffer and time stamps
        self.buffer         = self.prep_buffer(self.num_channels, self.buffer_length)
        self.time_stamps    = self.prep_time_stamps(self.buffer_length)

        self.softstate      = 'enabled' # For forcing/pausing stimulations


    def search_device(self):
        # =================================================================
        # Scans all ports and fills av_ports with valid serial COM values
        # INPUT
        #   /
        # OUTPUT
        #   av_ports        = Dictionnary of COM values for connection 
        #                     types (object)
        # =================================================================

        self.av_ports       = {'USB': None, 'BT': None}

        # Look for device: Bluetooth if available and USB only as fallback
        # -----------------------------------------------------------------
        myports = [tuple(ports) for ports in list(serial.tools.list_ports.comports())]
        
        for iPort in range(len(myports)):
            print(list(myports[iPort]))
            if '7&74D8485&0&000000000000_00000006' in list(myports[iPort])[2]:
                continue
                # # Bluetooth device query
                # self.av_ports["BT"] = list(myports[iPort])[0]
                # print('Found Helment connected via Bluetooth')
            elif 'Silicon Labs CP210x USB to UART Bridge' in list(myports[iPort])[1]:
                # USB device query
                self.av_ports["USB"] = list(myports[iPort])[0]
                print('Found Helment connected via USB')

        if all(val == None for val in list(self.av_ports.values())):
            raise Exception('Device could not be found')
        else:
            print(self.av_ports)


    def connect_board(self):
        # =================================================================
        # Connects to device, preferrably by USB
        # INPUT
        #   baud_rate       = baud rate of the port (scalar)
        #   time_out        = how long a message should be waited for
        #                     default = None (infinite), scalar or None
        #   av_ports        = Dictionnary with values being either None or 
        #                     a strong a the port to connect to
        # OUTPUT
        #   ser             = Serial connection (object)
        # =================================================================

        # Open communication protocol
        self.ser            = serial.Serial()
        self.ser.baudrate   = self.baud_rate
        self.ser.timeout    = self.time_out
        if self.av_ports["USB"] != None:
            self.ser.port   = self.av_ports["USB"]
            str_comtype = 'Communication via USB'
        elif self.av_ports["BT"] != None:
            self.ser.port   = self.av_ports["BT"]
            str_comtype = 'Communication via Bluetooth'
        print(str_comtype + ' established\n')

        
    def prep_buffer(self, num_channels, length):
        # This functions creates the buffer structure
        # that will be filled with eeg datasamples
        return np.zeros((num_channels, length))


    def prep_time_stamps(self, length):
        return np.zeros(length)


    def bin_to_voltage(self, bin):
        # =================================================================
        # Convert binary into volts
        # =================================================================
        #bin = int(bin) # requieres int

        if bin == 0:
            return 0
        if bin > 0 and bin <= 8388607:
            sign_bit = bin
        elif bin > 8388607 and bin <= 2*8388607:
            sign_bit = -2*8388607 + bin - 1
        
        voltage = (4.5*sign_bit)/(self.pga*8388607.0)
        voltage = voltage * 1000000 # Convert to microvolts
        return voltage


    def get_sample(self):
        # Get eeg samples from the serial connection
        raw_message = str(self.ser.readline())

        idx_start           = raw_message.find("{")
        idx_stop            = raw_message.find("}")
        raw_message         = raw_message[idx_start:idx_stop+1]
        
        # Handle JSON samples and add to signal buffer ----------------
        eeg_data_line       = json.loads(raw_message)

        # Each channel carries self.s_per_buffer amounts of samples
        buffer_line = [eeg_data_line['c1'],eeg_data_line['c2']]
        eeg_data        = np.array([buffer_line])
        eeg_data        = np.transpose(eeg_data)

        # Convert binary to voltage values
        for iBin in range(eeg_data.size):
            eeg_data[iBin] = self.bin_to_voltage(eeg_data[iBin])

        # # Flip signals because OpenBCI streams data P - N pins which 
        # # corresponds to Reference - Electrode
        # eeg_data        = np.array([eeg_data]) * -1
        
        return eeg_data


    def get_time_stamp(self):
        return round(time.perf_counter() * 1000 - self.start_time, 4)


    def fill_buffer(self):

        desired_con = 2 # HARD-CODED: Force USB mode

        if desired_con == 2: # Bluetooth
            self.ser.port        = self.av_ports["USB"]
            s_per_buffer    = 1
            print('USB')
        elif desired_con == 3: # USB
            self.ser.port        = self.av_ports["BT"]
            s_per_buffer    = 10
            print('BT')

        # Open communication ----------------------------------------------
        if self.ser.port == None:
            raise Exception('Verify that desired connection type (USB or Bluetooth) are indeed available')
        else:
            self.ser.open()

        print('Board is booting up ...')
        time.sleep(5)
        self.ser.write(bytes(str(desired_con), 'utf-8'))

        board_booting = True
        while board_booting and not keyboard.is_pressed('c'):
            raw_message = str(self.ser.readline())
            if '{' in raw_message and '}' in raw_message:
                print('Fully started')
                board_booting = False


        # This functions fills the buffer in self.buffer
        # that later can be accesed to perfom the real time analysis
        while True:
            # Get samples
            sample      = self.get_sample()
            time_stamp  = self.get_time_stamp()

            # Transpose vector
            if sample.shape[0] == 1:
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