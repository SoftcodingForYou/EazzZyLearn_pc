import numpy as np
import parameters as p
from threading import Thread
import time
import scipy

class SleepWakeState():


    def __init__(self):
        # =================================================================
        # Initialize sleep and wake staging
        # -----------------------------------------------------------------
        # Important parameters for timing of the staging process
        # We intitialize a time stamp at start which will be used for
        # periodic execution of the staging processes:
        # - Time stamp of code start (self.time_stamp)
        # - How often the staging is done (self.satging_interval)
        # Important parameters for timing of the staging process
        # =================================================================
        self.use_muse_sleep_classifier  = p.USE_MUSE_SLEEP_CLASSIFIER
        self.current_muse_metrics       = None
        self.muse_metric_map            = p.MUSE_METRIC_MAP
        self.last_sleep_stage_time      = 0
        self.last_wake_stage_time       = 0
        self.sleep_staging_interval     = p.SLEEP_STAGE_INTERVAL * 1000
        self.wake_staging_interval      = p.WAKE_STAGE_INTERVAL * 1000
        self.prior_awake                = np.ones(
            (1, int(p.LOCKING_LENGTH / p.WAKE_STAGE_INTERVAL)),
            dtype = 'bool') # Default last scores were awake = True
        self.sample_rate                = p.SAMPLERATE
        # Defaults
        self.isawake                    = True
        self.issws                      = False
        self.frequency_bands            = p.FREQUENCY_BANDS
        self.sleep_thresholds           = p.SLEEP_THRESHOLDS
        self.wake_thresholds            = p.WAKE_THRESHOLDS
         # This is neccesary to use is_alive() method
        self.sleep_thread               = Thread() 
        self.wake_thread                = Thread()


    def master_stage_wake_and_sleep(self, v_wake, v_sleep, freq_range, 
        output_file, time_stamp):
        # =================================================================
        # Method that checks whether it is time for staging sleep and/or 
        # wake. If so, the method calls for threads that predict said 
        # stages, store the infomration so that it is accessible by
        # real_time_algorithm() and save the information to disk.
        #
        #                               /!\
        # Calling two threads in parallel makes the code crash from time to
        # time (maybe when trying to access the same file at same time?). 
        # We prevent this hear by running one thread that calls wake and/or
        # sleep staging as necessary so that both of these sequentially and
        # therefore can not access output files at same time.
        # Conclusion:
        # It seems that the code crashed when estimating sleep stage during
        # power spectrum computation. We still leave this here as common 
        # thread, but the issue did not come from threading itself.
        #
        #                               /!\
        # 1. Important: reset timings outside of thread!
        # =================================================================        
        run_wake_staging                = False
        run_sleep_staging               = False
        if time_stamp >= ( self.last_wake_stage_time + 
            self.wake_staging_interval):
            self.last_wake_stage_time   = time_stamp
            run_wake_staging            = True
        if time_stamp >= ( self.last_sleep_stage_time + 
            self.sleep_staging_interval ):
            self.last_sleep_stage_time  = time_stamp
            run_sleep_staging           = True

        # Wake Thread
        if run_wake_staging == True:
            if not self.wake_thread.is_alive(): # If it is not alive, create a new one
                self.wake_thread = Thread(name='stage_wake_thread',
                    target=self.stage_wake_thread, 
                    args=(v_wake, freq_range, output_file, time_stamp))
                self.wake_thread.start()
            else:
                print('Skipped wake staging: Wake thread still active')

        # Sleep Thread
        if run_sleep_staging == True:
            if not self.sleep_thread.is_alive(): # If it is not alive, create a new one
                self.sleep_thread = Thread(name='stage_sleep_thread',
                    target=self.stage_sleep_thread, 
                    args=(v_sleep, freq_range, output_file, time_stamp))
                self.sleep_thread.start()
            else:
                print('Skipped sleep staging: Sleep thread still active')


    def stage_wake_thread(self, v_wake, freq_range, output_file, time_stamp):
        # =================================================================
        # Call wake staging methods. The sleep timer after the computations
        # is what prevent sthe code from breaking (specifically during 
        # sleep staging).
        # =================================================================
        self.staging(v_wake, 'isawake', freq_range, output_file, time_stamp)


    def stage_sleep_thread(self, v_sleep, freq_range, output_file, time_stamp):
        # =================================================================
        # Call sleep staging methods. The sleep timer after the 
        # computations is what prevent sthe code from breaking 
        # (specifically during sleep staging).
        # =================================================================
        self.staging(v_sleep, 'issws', freq_range, output_file, time_stamp)


    def staging(self, v_wake, staging_what, freq_range, output_file,
        time_stamp):

        predictions             = {}
        # Get ratio thresholds of interest based on sleep or wake staging
        if staging_what == 'isawake':
            staging_ratios  = self.wake_thresholds
            line_base       = str(time_stamp) + ', Subject awake = '
        elif staging_what == 'issws':
            staging_ratios  = self.sleep_thresholds
            line_base       = str(time_stamp) + ', Subject in SWS = '
        staging_ratios_keys = list(staging_ratios.keys())

        if not self.use_muse_sleep_classifier:
            v_power, v_freqs    = self.power_spectr_welch(v_wake,
                freq_range, self.sample_rate)

            # Get the stage prediction from different band ratios
            for iRatio in range(len(staging_ratios_keys)):
                
                # Define paramaters (number/name of bands to compare, thresholds)
                ratio_str       = staging_ratios_keys[iRatio]
                ratio_thresold  = staging_ratios[ratio_str]
                bands           = ratio_str.split("VS") # Has to generate a list
            
                predictions[ratio_str] = self.band_ratio_thresholding(v_power,
                                    v_freqs, bands, ratio_thresold)

            # Take decision based on individual predictions and update workspace
            indiv_predictions       = list(predictions.values())
            if staging_what == 'isawake':
                # if any(prediction == True for prediction in predictions.values()): # Seems too strict
                # We take the average prediction as the final decision
                if sum(indiv_predictions) >= len(indiv_predictions) / 2:
                    self.isawake    = True
                    line_add        = True
                else:
                    self.isawake    = False
                    line_add        = False

                # Check for false positive in awake staging because of movement 
                # artifacts
                # Update last awake stagings
                self.prior_awake    = np.append(self.prior_awake, self.isawake)
                self.prior_awake    = np.delete(self.prior_awake, 0)
                # Get overall staging value: If >= 50% predicted awake = True,
                # then we lock current stage into being awake
                if ( np.sum(self.prior_awake) >= self.prior_awake.size / 2 and
                    self.isawake == False ):
                    self.isawake    = True
                    line_add        = str(line_add) + ' (False negative)'
                    
            elif staging_what == 'issws':
                # We take the average prediction as the final decision
                if sum(indiv_predictions) >= len(indiv_predictions) / 2:
                # if any(prediction == True for prediction in predictions.values()): # False positives with old values
                    self.issws      = True
                    line_add        = True
                else:
                    self.issws      = False
                    line_add        = False
        elif self.current_muse_metrics is not None:
            for key, value in self.muse_metric_map.items():
                predictions[key] = self.current_muse_metrics[value]

            is_unknown = len([val for val in predictions.values() if val != 0]) == 0 # Unknown when all stages are zeros
            max_key = max(predictions, key=predictions.get)
            if staging_what == 'issws':
                if is_unknown or max_key != 'N3':
                    self.issws = False
                    line_add = False
                else:
                    self.issws = True
                    line_add = True
            elif staging_what == 'isawake':
                if is_unknown or max_key != 'Wake':
                    self.isawake = False
                    line_add = False
                else:
                    self.isawake = True
                    line_add = True
        else:
            return

        # Store staging on disk
        # -----------------------------------------------------------------
        predictions = self._convert_numpy_values(predictions)
        line = line_base + str(line_add) + ' ' + str(predictions)
        self.stage_write(line, output_file)


    def _convert_numpy_values(self, predictions):
        """Convert numpy values in predictions dict to native Python types"""
        converted = {}
        for key, value in predictions.items():
            if hasattr(value, 'item'):  # numpy scalar
                converted[key] = value.item()
            elif isinstance(value, np.ndarray):  # numpy array
                converted[key] = value.tolist()
            else:
                converted[key] = value
        return converted


    def band_ratio_thresholding(self, power, freqs, bands, threshold_val):
        # =================================================================
        # Very simple method that calculates the ratios between two 
        # frequency bands.
        # power = array of power (from sp.MultiTapering = outcome.psd)
        # freqs = array of frequencies (Hz)
        # bands = list of strings pointing to bands_dict entries
        # threshold_val = scalar of threshold power (ratio) values
        # It is probably best to call this method several times and to then
        # compare if all calls result in the same sleep stage in the main 
        # process
        # =================================================================

        prediction = False # Default

        if len(bands) == 1:

            # Working directly with band power values
            band = bands[0]

            # Get frequency edges
            passband            = self.frequency_bands[band]
            
            low                 = np.where(freqs < passband[1])[0]
            high                = np.where(freqs > passband[0])[0]
            f_pass              = np.in1d(high, low)
            overlap             = high[np.where(f_pass == True)[0]]
            
            band_power          = np.mean(power[overlap])

            if band_power > threshold_val:
                prediction = True

        elif len(bands) == 2:

            # Working with the ratio between band power values
            band_power          = np.zeros((2, 1))
            for iBand in range(len(bands)):
                # Get frequency edges
                passband        = self.frequency_bands[bands[iBand]]
                
                low             = np.where(freqs < passband[1])[0]
                high            = np.where(freqs > passband[0])[0]
                f_pass          = np.in1d(high, low)
                overlap         = high[np.where(f_pass == True)[0]]
                
                band_power[iBand]= np.mean(power[overlap])

            band_ratio          = band_power[0] / band_power[1]
            band                = bands[0] + bands[1]

            if band_ratio > threshold_val:
                prediction = True

        else:
                raise Exception('Too many elements for freq band list')

        return prediction


    def stage_write(self, line, output_file):
        # =================================================================
        # Store on disk the stage information
        # Note here that only when calling close(), the information gets
        # indeed written into the file.
        # =================================================================
        with open(output_file, 'a') as f: # Appending
            f.write(line + '\n')
        print(line[0:line.find(' {')]) # For us to follow


    def power_spectr_welch(self, whole_range_signal, freq_range, sample_rate):

        win = 4 * sample_rate # Lowest frequency of interest = 0.5 --> 2x low signal in window is optimal
        freqs, power    = scipy.signal.welch(whole_range_signal,
            sample_rate, nperseg=win)
        
        Fpass           = (freq_range[0], freq_range[1])
        f_highpass      = np.where(freqs >= Fpass[0])[0]
        f_lowpass       = np.where(freqs <= Fpass[-1])[0]
        f_passband      = np.in1d(f_highpass, f_lowpass)
        findx           = f_highpass[np.where(f_passband == True)[0]]
        
        power           = power[findx]
        freqs           = freqs[findx]

        return power, freqs