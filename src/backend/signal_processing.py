import scipy.signal
import numpy            as np
import parameters       as p


class SignalProcessing():


    def __init__(self):
        # =================================================================
        # Intialize signal processing
        # -----------------------------------------------------------------
        # One of the most important classes that contains everything 
        # related to the processing of the input buffer.
        # We initialize:
        # - Channel extraction information
        # - Filter parameters such as order and window method
        # - Filter coefficients b and a
        # - Frequency ranges
        # - Lengths (in samples) of the arrays for different purposes:
        #    * main for filtering; the longer the buffer, the more 
        #      accurate the filter output)
        #    * SO for slow oscillation prediction)
        #    * threshold for thresholding of downstates in delta
        #    * sleep and wake for determining current stages. The wake 
        #      response buffer is short so that wake patterns are detected 
        #      as early as possible
        #    * Padding signal lengths
        # =================================================================

        self.channel                    = p.IDX_ELEC
        self.reference_channel          = p.IDX_REF
        self.channels                   = list(p.ELEC.keys())
        
        filt_order                      = p.FILT_ORDER

        # These frequencies will be used by other functions
        self.freq_range_whole           = p.FREQUENCY_BANDS["Whole"]
        self.freq_range_delta           = p.FREQUENCY_BANDS["Delta"]
        self.freq_range_slowdelta       = p.FREQUENCY_BANDS["SlowDelta"]
        self.freq_range_stopband        = p.FREQUENCY_BANDS["LineNoise"]

        # Bandpass filter whole signal
        self.b_wholerange, self.a_wholerange = scipy.signal.butter(
            filt_order, self.freq_range_whole,
            btype='bandpass', fs=p.SAMPLERATE)

        # Delta range (0.5 - 4 Hz)
        self.b_delta, self.a_delta      = scipy.signal.butter(
            filt_order, self.freq_range_delta,
            btype='bandpass', fs=p.SAMPLERATE)
        
        # Slow delta range (0.5, 2 Hz)
        self.b_slowdelta, self.a_slowdelta = scipy.signal.butter(
            filt_order, self.freq_range_slowdelta,
            btype='bandpass', fs=p.SAMPLERATE)

        # Stopband filter electrical noise
        self.b_notch, self.a_notch      = scipy.signal.butter(
            filt_order, self.freq_range_stopband, btype='stop',
            fs=p.SAMPLERATE)

        # Information about array lengths
        self.main_buffer_length         = p.MAIN_BUFFER_LENGTH
        self.delta_buffer_length        = p.DELTA_BUFFER_LENGTH
        self.threshold_buffer_length    = p.THRESHOLD_BUFFER_LENGTH
        self.sleep_buffer_length        = p.SLEEP_BUFFER_LENGTH
        self.wake_buffer_length         = p.REPONSE_BUFFER_LENGTH

        if ( self.main_buffer_length < self.threshold_buffer_length or
            self.main_buffer_length < self.sleep_buffer_length or
            self.main_buffer_length < self.wake_buffer_length ):
            raise Exception('Processing arrays cannot be shorter than the main buffer length')

        # Determine padding length for signal filtering (eliminating edge 
        # artifacts is crucial in our application)
        default_pad                     = 3 * max(len(self.a_wholerange), 
                                                len(self.b_wholerange))
        if default_pad > self.main_buffer_length/10-1:
            self.padlen                 = int(default_pad) # Scipy expects int
            print('Default padding length chosen: 3 * max length of filter coefficients')
        else:
            self.padlen                 = int(self.main_buffer_length/10-1) # Scipy expects int
        # self.padlen                     = int(self.main_buffer_length-1)


    def filt_signal_filtfilt(self, signal, b, a):
        # =================================================================
        #
        #           /!\ This method seems to have the highest
        #             impact on the code execution speed /!\
        #
        # This method filters the signal. Important here is to achieve 
        # least amount of distortion of the signal as possible. Phase 
        # shifts are NOT acceptable in this context.
        # This method should be performed on the main buffer array 
        # (longest) before splitting array into process-specific arrays.
        # This method is also suitable for applying notch filter, but 
        # Notch filtering should only be applied if disabled in OpenBCI GUI
        # 
        # FiltFilt is NOT RECOMMENDED FOR ONLINE applications!
        # =================================================================
        signal_filtered    = scipy.signal.filtfilt(b, a, signal, 
                                padtype='odd', padlen=self.padlen)

        return signal_filtered


    def filt_signal_online(self, signal, b, a):
        # =================================================================
        #
        #           /!\ This method seems to have the highest
        #             impact on the code execution speed /!\
        #
        # This method filters the signal. Important here is to achieve 
        # least amount of distortion of the signal as possible. Phase 
        # shifts are NOT acceptable in this context.
        # This method should be performed on the main buffer array 
        # (longest) before splitting array into process-specific arrays.
        # This method is also suitable for applying notch filter, but 
        # Notch filtering should only be applied if disabled in OpenBCI GUI
        # Linear phase filters are suitable for online uses but introuce 
        # time delays.
        # =================================================================
        padded_signal   = np.pad(signal, (self.padlen, 0), 'symmetric')
        signal_filtered = scipy.signal.lfilter(b, a, padded_signal)
        signal_filtered = signal_filtered[self.padlen+1:]

        return signal_filtered

    
    def extract_array(self, array, length):
        # =================================================================
        # This method is responsable for building the arrays from the main 
        # buffer array. This is crucial since all arrays for the different 
        # purposes are of distinct lengths.
        # array             = numpy vector, must be 1D
        # length            = scalar, length in samples
        # anticipated_end   = scalar, length in samples where the end
        #                     defines whether the signal is extracted from 
        #                     inside the array instead of from the end 
        # =================================================================
        return array[-length:]


    def master_extract_signal(self, buffer):
        """
        Extracts and filters EEG (or similar biosignal) data from a multi-channel buffer 
        into multiple frequency-specific signal arrays.

        This method takes a multi-channel time-series buffer and performs a sequence 
        of filtering operations to remove noise and isolate relevant frequency bands. 
        It returns versions of the signal suitable for wake and sleep analyses, as well 
        as delta and slow-delta filtered signals.

        Steps:
            1. Selects the raw signal from the specified channel (optional re-referencing).
            2. Removes electrical noise via a notch filter.
            3. Bandpass filters the signal into a usable range (e.g., 0.1 - 45 Hz).
            4. Applies online (causal) filters to extract delta and slow-delta bands.
            5. Extracts arrays of appropriate lengths for wake/sleep processing.

        Args:
            buffer (np.ndarray):
                2D array of shape (n_channels, n_samples) containing the recorded 
                biosignal data. Each row corresponds to one channel, and each column 
                to a time sample.

        Returns:
            tuple:
                (
                    v_wake (np.ndarray): 
                        Bandpass-filtered signal (0.1 - 45 Hz) trimmed to wake buffer length.
                    v_sleep (np.ndarray): 
                        Bandpass-filtered signal (0.1 - 45 Hz) trimmed to sleep buffer length.
                    v_filtered_delta (np.ndarray): 
                        Delta-band filtered signal (untrimmed).
                    v_delta (np.ndarray): 
                        Delta-band signal trimmed to delta buffer length.
                    v_slowdelta (np.ndarray): 
                        Slow-delta signal trimmed to delta buffer length.
                    v_clean_filtfilt (np.ndarray): 
                        Notch-filtered signal with phase-preserving (filtfilt) filtering.
                )

        Notes:
            - The filtering uses both zero-phase (`filtfilt`) and online (causal) filters. As such,
                the delta and slow-delta signals are suitable for real-time applications, while the
                notch-filtered signals are not!
            - NaN interpolation is commented out but may be used if missing samples occur.
        """
        v_raw           = buffer[self.channel, :]

        if self.reference_channel > -1:
            v_ref       = buffer[self.reference_channel, :]
            v_raw       = v_raw - v_ref


        # nans, x = self.nan_helper(v_raw)
        # v_raw[nans]= np.interp(x(nans), x(~nans), v_raw[~nans])

        # Reject electrical noise:
        v_clean_filtfilt      = self.filt_signal_filtfilt(v_raw, 
            self.b_notch, self.a_notch)
        # Banspass filter into whole freq range [0.1, 45] Hz
        v_filtered_whole      = self.filt_signal_filtfilt(v_clean_filtfilt, 
            self.b_wholerange, self.a_wholerange)

        # We use filt_signal_online since it only shifts the signal by one
        # sample
        v_notched_online      = self.filt_signal_online(v_raw, 
            self.b_notch, self.a_notch)
        # Filter into delta ranges
        v_filtered_delta      = self.filt_signal_online(v_notched_online, 
            self.b_delta, self.a_delta)
        v_filtered_slowdelta  = self.filt_signal_online(v_notched_online, 
            self.b_slowdelta, self.a_slowdelta)

        # Extract vectors of correct lengths
        v_wake           = self.extract_array(v_filtered_whole,
            self.wake_buffer_length)
        v_sleep          = self.extract_array(v_filtered_whole,
            self.sleep_buffer_length)
        v_delta         = self.extract_array(v_filtered_delta,
            self.delta_buffer_length)
        v_slowdelta     = self.extract_array(v_filtered_slowdelta,
            self.delta_buffer_length)

        return v_wake, v_sleep, v_filtered_delta, v_delta, v_slowdelta
    

    def nan_helper(self, y):
        """Helper to handle indices and logical indices of NaNs.

        Input:
            - y, 1d numpy array with possible NaNs
        Output:
            - nans, logical indices of NaNs
            - index, a function, with signature indices= index(logical_indices),
            to convert logical indices of NaNs to 'equivalent' indices
        """
        return np.isnan(y), lambda z: z.nonzero()[0]


    def switch_channel(self, number_pressed, outputfile, timestamp):
        # =================================================================
        # This method allows us to switch between channels for slow 
        # oscillation detection and staging on the fly if the current one 
        # for example goes bad.
        # =================================================================
        idx_channel = int(number_pressed) - 1 # -1 for Python indexing
        if idx_channel != self.channel:
            self.channel = idx_channel
            line = str(timestamp) + ', Switched channel to ' + str(number_pressed) + ' (' + self.channels[self.channel] +')'
            stimhistory = open(outputfile, 'a') # Appending
            stimhistory.write(line + '\n')
            stimhistory.close()
            print(line)
        else:
            print('Channel already set to ' + number_pressed + ' (' + self.channels[self.channel] +')')


    def switch_online_reference_channel(self, number_pressed, outputfile, timestamp):
        idx_channel = int(number_pressed) - 1 # -1 for Python indexing
        if idx_channel != self.reference_channel:
            self.reference_channel = idx_channel
            if self.reference_channel == -1:
                line = str(timestamp) + ', Switched online reference to None'
            else:
                line = str(timestamp) + ', Switched online reference to ' + str(number_pressed) + ' (' + self.channels[self.reference_channel] +')'
            stimhistory = open(outputfile, 'a') # Appending
            stimhistory.write(line + '\n')
            stimhistory.close()
            print(line)
        else:
            print('Online reference already set to ' + number_pressed + ' (' + self.channels[self.reference_channel] +')')
