import numpy as np
import scipy
import scipy.signal
import parameters as p
import numpy as np
from threading import Thread



class PredictSlowOscillation:

    def __init__(self):
        # =================================================================
        # Initialize slow oscillation predictions
        # -----------------------------------------------------------------
        # We set default states and threshold values for the signal to 
        # satisfy in order to be considered a valid SO downstate so that we
        # then call for prediction of SO upstate
        # =================================================================
        
        # Checkpoint parameters initialization
        self.stim_at_stamp          = None
        
        self.default_threshold      = p.DEFAULT_THRESHOLD
        self.artifact_threshold     = p.NON_PHYSIOLOGICAL_THRESHOLD

        self.throw_multi            = p.THROW_MULTIPLICATION


    def set_threshold(self, threshold_array):
        # =================================================================
        # This method is setting threshold values based on the overall
        # signal amplitude.
        # =================================================================
        v_envelope                  = np.absolute(scipy.signal.hilbert(threshold_array))
        adaptive_threshold          = - np.mean(threshold_array) - 1.1 * np.std(v_envelope) 

        return adaptive_threshold


    def extract_slow_oscillation_onset(self, delta_array, slow_delta_array):
        # =================================================================
        # This method finds an ongoing slow oscillation candidate inside an
        # array. We define an ongoing slow oscillation by a pos-to-neg 
        # zero-crossing that has not yet reached the minimum amplitude 
        # (slow oscillation negative peak).
        # =================================================================

        # Determine last positive-to-negative zero-crossing
        sign_vector             = np.sign(delta_array)
        diff_vector             = np.diff(sign_vector) # -2 = pos2neg cross
        idx_p2n_all             = np.where(diff_vector == -2)[0]
        if len(idx_p2n_all) == 0:
            return None, None, None
        else:
            idx_p2n             = idx_p2n_all[-1]
            # Last zero-crossing is separating our signal of interest (to 
            # the right) from the thresholding signal (to left)

        # Verify that we are still in the downstate phase
        idx_n2p_all             = np.where(diff_vector == 2)[0]
        if len(idx_n2p_all) != 0 and idx_n2p_all[-1] > idx_p2n:
            return None, None, None # We are already past the downphase and 
                                    # should ignore this slow oscillation

        return delta_array[idx_p2n:], slow_delta_array[idx_p2n:], idx_p2n


    def downstate_validation(self, SO_onset_array, threshold):
        # =================================================================
        # This method is looking for a downstate of potential SOs:
        # We can only be sure we are looking at a downstate if the sample
        # of minimum amplitude (downstate) is followed by a sample of an
        # amplitude which goes up again.
        # Valid downstates bypass an amplitude threshold that is 
        # online-adapted by self.set_threshold().
        # Amplitudes below -300 uV are considered non-physiological.
        # =================================================================
        idx_downstate           = np.argmin(SO_onset_array)
        post_down_signal        = SO_onset_array[idx_downstate:]
        if len(post_down_signal) < 2:
            # Here, we assure that we have indeed reached a downstate, 
            # whereafter the signal goes up again
            downstate_valid = False
            return downstate_valid
        elif (post_down_signal[0] < threshold and
            post_down_signal[0] < post_down_signal[1] and 
            post_down_signal[0] > self.artifact_threshold):
            downstate_valid = True
        else:
            downstate_valid = False
        return downstate_valid


    def multiply_throw_time(self, onset_SO, sampling_rate, down_time):
        # =================================================================
        # This method considers a perfect sine wave and predicts the 
        # upstate as 3 times the time it took the delta signal to go from 
        # zero to downstate.
        # =================================================================
        samples_down_to_up      = self.throw_multi * onset_SO.size
        time_down_to_up         = (samples_down_to_up / sampling_rate) * 1000
        stim_time_stamp         = down_time + time_down_to_up

        return stim_time_stamp


    def correct_stim_time(self, stim_time, cue_duration):
        delta_stim_time     = cue_duration / 3
        stim_time           = stim_time - (2 * delta_stim_time) # 2 third before upstate
        return stim_time


    def timestamp_downstate(self, SO_onset, current_time, sample_rate):
        # -----------------------------------------------------------------
        # During this class execution, we might be way past a downstate 
        # already and therefore, the downstate time stamp should not be the
        # current time, but should instead be determined
        # -----------------------------------------------------------------
        number_samples      = SO_onset.size - 1 # -1 because of Python indexing
        down_sample         = np.argmin(SO_onset)
        time_shift          = round(
            (number_samples - down_sample) * 1000 / sample_rate, 4)
        downstate_timestamp = current_time - time_shift
        return downstate_timestamp


    def predicted_SO_write(self, line, output_file):
        # =================================================================
        # Store on disk the information about predicted SO upstate times.
        # Note here that only when calling close(), the information gets
        # indeed written into the file.
        # =================================================================
        self.predhistory = open(output_file, 'a') # Appending
        self.predhistory.write(line + '\n')
        self.predhistory.close()


    def predicted_SO_write_thread(self, line, output_file):
        self.predicted_SO_write(line, output_file)


    def master_slow_osc_prediction(self, uncut_delta, delta, slowdelta, 
        length_threshold, sample_rate, current_time, cue_duration, 
        predicted_SO_path):
        # =================================================================
        # Method grouping together all necessary steps to predict slow osc.
        # upstates. We will build delta and slow delta vectors,
        # which respectively, we define thresholds and predict the sine 
        # wave on. (Slow) Delta signals are extract after their onset, 
        # everything befroe will be rejected and only serves for the 
        # threshold buffer. 
        # =================================================================

        # Extract the signal at first slow oscillation onset sample
        on_delta, on_sldelta, idx_p2n = self.extract_slow_oscillation_onset(
            delta, slowdelta)

        if idx_p2n is None:
            return

        # Adaptive threshold
        threshold   = self.set_threshold(uncut_delta)
        
        # Validate slow oscillation downstate
        valid_downstate = self.downstate_validation(on_delta, threshold)
        if valid_downstate == False:
            return
        else:
            downstate_time = self.timestamp_downstate(on_delta, 
                current_time, sample_rate)
            
        # Predict slow oscillations upstate
        stim_at_stamp = self.multiply_throw_time(on_delta,
            sample_rate, downstate_time)

        if stim_at_stamp == None:
            return
        else:
            self.stim_at_stamp = self.correct_stim_time(stim_at_stamp,
                cue_duration)

            # We store the upstate time stamp (non-corrected!)
            line = str(downstate_time) + ', Predicted upstate at ' + str(stim_at_stamp)
            self.pred_thread = Thread(name='write_prediction_thread',
                target=self.predicted_SO_write_thread, 
                args=(line, predicted_SO_path))
            self.pred_thread.start()