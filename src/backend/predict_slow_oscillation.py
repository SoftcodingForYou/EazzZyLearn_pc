import numpy            as np
import scipy
import scipy.signal
import parameters       as p
from backend.disk_io    import DiskIO


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

        self.downstate_threshold    = p.DEFAULT_THRESHOLD
        self.artifact_threshold     = p.NON_PHYSIOLOGICAL_THRESHOLD
        self.sd_multi               = p.SD_MULTIPLICATOR
        self.last_threshold_update  = 0

        # Adaptive trough multiplication coefficient (starts at 1.25)
        # This coefficient is continuously updated based on actual vs predicted timing
        self.trough_multi           = p.TROUGH_MULTIPLICATION
        self.trough_multi_history   = [p.TROUGH_MULTIPLICATION] * 100  # Rolling buffer for averaging

        # Validation state tracking
        self.samples_down_to_up_predicted = 0  # Store prediction for validation
        self.is_valid_prediction    = False    # Flag for valid predictions
        self.downstate_amplitude_valid = False # Track downstate amplitude validity
        self.is_positive_half_wave  = False    # Track if we're in positive half-wave

        # Timing bounds for physiological slow oscillations (0.5-2 Hz)
        # Min: 125ms (corresponds to 2Hz), Max: 1000ms (corresponds to 0.5Hz)
        self.time_down_to_up_edges  = [125, 1000]  # milliseconds

        self.disk_io                = DiskIO(
            p.MAX_BUFFERED_LINES, p.PREDICTION_FLUSH_INTERVAL, 'slow_osc_thread')
        self.last_downstate_amplitudes = [p.DEFAULT_THRESHOLD] * 3
        self.is_same_downstate = False


    def set_threshold(self, threshold_array, current_time, is_fast):
        """
        Calculate adaptive threshold for slow oscillation detection using Hilbert envelope.

        Uses the Hilbert transform to extract the smooth amplitude envelope of the signal,
        then calculates a threshold based on the envelope's statistics. This provides
        phase-independent amplitude estimation crucial for detecting genuine slow waves.

        Args:
        threshold_array (np.ndarray):
            EEG signal array (typically 30s of slow delta band 0.5-2Hz)

        Returns:
            adaptive_threshold (float):
                Negative threshold value in microvolts. Downstates must exceed this
                threshold (be more negative) to be considered valid slow oscillations.
                Formula: -(mean + sd_multiplicator * std) of the Hilbert envelope

        Notes:
            The Hilbert transform provides advantages over simple absolute values:
            - Smooth envelope tracking oscillation strength, not instantaneous values
            - Better stability for low-frequency (0.5-2Hz) slow oscillations
            - Reduces false positives from noise fluctuations
            - Preserves physiological characteristics of sleep slow waves
        """

        if current_time - self.last_threshold_update < 500:
            return;

        if is_fast:
            analytic_signal     = threshold_array
        else:
            analytic_signal     = scipy.signal.hilbert(threshold_array)

        v_envelope              = np.absolute(analytic_signal)
        self.downstate_threshold = - np.mean(v_envelope) - self.sd_multi * np.std(v_envelope)
        self.last_threshold_update = current_time


    def set_artifact_threshold(self, trough_amplitude):
        """
        Update adaptive artifact rejection threshold.

        Updates the rolling buffer of last 3 downstate amplitudes and calculates
        a new artifact threshold as 2.5x the mean of recent downstates.

        Args:
            trough_amplitude (float): Amplitude of current validated downstate (µV)

        Sets:
            float: New artifact threshold (typically 2.5× mean of recent downstates)
        """
        self.last_downstate_amplitudes[:-1] = self.last_downstate_amplitudes[1:]
        self.last_downstate_amplitudes[-1] = trough_amplitude
        self.artifact_threshold = np.mean(self.last_downstate_amplitudes) * 2.5


    def extract_slow_oscillation_onset(self, delta_array):
        """
        Find the most recent slow oscillation onset (positive-to-negative zero crossing).

        Detects the most recent downstate onset by finding the last point where
        the delta signal crosses from positive to negative, and verifies we are
        still in the negative half-wave (haven't crossed back to positive yet).

        Args:
            delta_array (np.ndarray): Delta-filtered signal array (5s buffer)

        Returns:
            tuple: (on_delta, idx_p2n)
                on_delta: Signal from onset point onward (empty array if no valid onset)
                idx_p2n: Index of positive-to-negative crossing (-1 if no valid onset)
        """

        idx_p2n = -1
        on_delta = np.array([])

        # Determine last positive-to-negative zero-crossing
        sign_vector             = np.sign(delta_array)
        diff_vector             = np.diff(sign_vector)

        # Find all positive-to-negative crossings (-2)
        idx_p2n_all             = np.where(diff_vector == -2)[0]

        if len(idx_p2n_all) == 0:
            return on_delta, idx_p2n
        else:
            idx_p2n             = idx_p2n_all[-1]
            on_delta            = delta_array[idx_p2n:]
            # Last zero-crossing is separating our signal of interest (to
            # the right) from the thresholding signal (to left)

        # Verify that we are still in the negative halfwave
        idx_n2p_all             = np.where(diff_vector == 2)[0]

        if len(idx_n2p_all) > 0 and idx_n2p_all[-1] > idx_p2n:
            self.is_same_downstate = False
            self.is_positive_half_wave = True
        else:
            self.is_positive_half_wave = False

        return on_delta, idx_p2n


    def downstate_validation(self, SO_onset_array):
        """
        Validate detected downstate using multiple criteria.

        This method looks for a downstate of potential slow oscillations.
        We can only be sure we are looking at a downstate if the sample of minimum
        amplitude (downstate) is followed by a sample of an amplitude which goes up
        again. Valid downstates bypass an amplitude threshold that is online-adapted
        by set_threshold(). Amplitudes below -300 µV are considered non-physiological.

        IMPROVED: Now processes downstate exactly once (when post_down_length == 2)
        to prevent repeated validation on the same downstate.

        Args:
            SO_onset_array (np.ndarray): Signal from p2n crossing onward
            threshold (float): Adaptive amplitude threshold (from set_threshold)

        Returns:
            bool: True if downstate is valid, False otherwise

        VALIDATION CRITERIA (all must pass):
            1. Length: At least 2 samples since trough (to verify upward trend)
            2. Amplitude: Trough below adaptive threshold (strong enough)
            3. Upward trend: Signal goes up after trough (we've reached minimum)
            4. Artifact: Trough above artifact threshold (physiologically plausible)
        """

        if len(SO_onset_array) == 0:
            self.downstate_amplitude_valid = False
            return False

        idx_downstate           = np.argmin(SO_onset_array)
        amp_downstate           = SO_onset_array[idx_downstate]
        post_down_length        = len(SO_onset_array) - idx_downstate

        # Amplitude validation (separate from timing validation)
        if (post_down_length > 0 and
            amp_downstate < self.downstate_threshold and
            amp_downstate > self.artifact_threshold):
            self.downstate_amplitude_valid = True
        else:
            self.downstate_amplitude_valid = False

        # Criteria: Length, Amplitude, upward trend, and artifact checks
        if (post_down_length == 2 and # Process only once
            self.downstate_amplitude_valid and
            amp_downstate < SO_onset_array[idx_downstate + 1]):

            # Update artifact threshold only on FIRST detection of this downstate
            if not self.is_same_downstate:
                self.set_artifact_threshold(amp_downstate)

            self.is_same_downstate = True
            return True
        else:
            return False


    def upstate_validation(self, slow_delta_onset_array):
        """
        Validate detected upstate and update adaptive trough multiplication coefficient.

        This function verifies that we have reached a true upstate (positive peak) after a 
        previously detected downstate. It compares the actual timing from downstate to upstate 
        against the predicted timing to adaptively update the trough multiplication coefficient used
        for future predictions.

        This enables personalized adaptation to individual slow oscillation morphology,
        improving prediction accuracy over time.

        Args:
            slow_delta_onset_array (np.ndarray): Slow delta signal from p2n crossing onward

        VALIDATION CRITERIA (all must pass):
            1. Downstate occurs before upstate
            2. Upstate is at the end of the array (processing once per upstate)
            3. Upstate amplitude is positive (true positive half-wave)
            4. Upstate amplitude exceeds the sample after downstate (confirmed peak)
            5. A valid downstate-to-upstate prediction exists (samples_down_to_up_predicted > 0)
            6. Prediction from downstate was valid (within physiological bounds)
            7. Valid downstate with respect to amplitude (not timing)
        """
        if len(slow_delta_onset_array) < 2:
            return

        idx_downstate = np.argmin(slow_delta_onset_array)
        idx_upstate = np.argmax(slow_delta_onset_array)

        # All 7 validation criteria must pass
        if (idx_downstate < idx_upstate and
            idx_upstate == len(slow_delta_onset_array) - 2 and  # Process once (upstate + 1 sample after)
            slow_delta_onset_array[idx_upstate] > 0 and  # True positive half-wave
            slow_delta_onset_array[idx_upstate] > slow_delta_onset_array[idx_downstate + 1] and
            self.samples_down_to_up_predicted > 0 and
            self.is_valid_prediction and
            self.downstate_amplitude_valid):

            # Calculate actual coefficient from this slow oscillation
            actual_samples = idx_upstate - idx_downstate
            actual_coefficient = actual_samples / self.samples_down_to_up_predicted

            # Update rolling coefficient history (shift and add new)
            self.trough_multi_history[:-1] = self.trough_multi_history[1:]
            self.trough_multi_history[-1] = actual_coefficient

            # Average last 3 coefficients for stability
            self.trough_multi = np.mean(self.trough_multi_history)


    def multiply_through_time(self, onset_SO, sampling_rate, down_time):
        """
        Predict upstate timestamp using adaptive sine wave model.

        IMPROVED: Now validates prediction against physiological timing bounds
        and stores prediction for later validation against actual upstate.

        This method considers a perfect sine wave and predicts the upstate as
        trough_multi times the time it took the delta signal to go from zero
        (p2n crossing) to downstate (trough).

        Args:
            onset_SO (np.ndarray):
                Delta signal from positive-to-negative zero-crossing onwards.
                The minimum (trough) indicates the downstate location.
            sampling_rate (float):
                EEG sampling rate in Hz
            down_time (float):
                Timestamp in milliseconds when downstate trough was detected

        Returns:
            float or None:
                Predicted upstate timestamp in milliseconds (absolute time).
                Calculated as: down_time + (trough_multi * time_to_trough)
                Returns None if prediction is outside physiological bounds.
        """

        # Calculate samples from zero-crossing to trough
        samples_to_trough       = np.argmin(onset_SO) + 1

        # Store predicted samples for later validation
        self.samples_down_to_up_predicted = self.trough_multi * samples_to_trough

        # Convert samples to time in milliseconds
        time_down_to_up         = (self.samples_down_to_up_predicted / sampling_rate) * 1000
        stim_time_stamp         = down_time + time_down_to_up

        # Validate prediction is within physiological bounds (125-1000ms for 0.5-2Hz)
        if time_down_to_up < self.time_down_to_up_edges[0] or \
           time_down_to_up > self.time_down_to_up_edges[1]:
            # Invalid prediction, outside acceptable range
            self.is_valid_prediction = False
        else:
            self.is_valid_prediction = True
            
        return stim_time_stamp


    def correct_stim_time(self, stim_time, cue_duration):
        """
        Shift stimulation timing to optimize cue delivery.

        Adjusts the predicted upstate timestamp to account for cue audio duration.
        The cue should play 2/3 before the upstate peak and 1/3 during/after it,
        maximizing the overlap between audio and the upstate for optimal TMR effect.

        Args:
            stim_time (float): Predicted upstate timestamp (milliseconds)
            cue_duration (float): TMR audio cue length (milliseconds)

        Returns:
            float: Corrected stimulation timestamp (milliseconds)

        Algorithm:
            Shifts timing back by 2/3 of cue duration so that:
            - 2/3 of cue plays before upstate peak
            - 1/3 of cue plays during/after upstate peak

        Note:
            The corrected time can fall into the past depending on cue length
        """
        delta_stim_time     = cue_duration / 3
        stim_time           = stim_time - (2 * delta_stim_time) # 2 third before upstate
        return stim_time


    def timestamp_downstate(self, SO_onset, current_time, sample_rate):
        """
        Calculate actual timestamp of downstate trough.

        During this function execution, we might be way past a downstate already
        and therefore, the downstate timestamp should not be the current time, but
        should instead be determined by accounting for processing delay.

        Args:
            SO_onset (np.ndarray): Signal from p2n crossing onward
            current_time (float): Current sample timestamp (milliseconds)
            sample_rate (float): EEG sampling rate (Hz)

        Returns:
            float: Timestamp when the trough actually occurred (milliseconds)
        """
        number_samples      = SO_onset.size - 1 # -1 because of Python indexing
        down_sample         = np.argmin(SO_onset)
        time_shift          = (number_samples - down_sample) * 1000 / sample_rate
        downstate_timestamp = current_time - time_shift
        return downstate_timestamp


    def master_slow_osc_prediction(self, uncut_delta, delta, slowdelta,
        length_threshold, sample_rate, current_time, cue_duration,
        predicted_SO_path):
        """
        Master method for slow oscillation upstate prediction.

        IMPROVED: Now includes upstate validation phase for adaptive learning
        of the trough multiplication coefficient.

        Method grouping together all necessary steps to predict slow osc.
        upstates. We will build delta and slow delta vectors, which respectively,
        we define thresholds and predict the sine wave on. (Slow) Delta signals
        are extract after their onset, everything before will be rejected and only
        serves for the threshold buffer.

        ALGORITHM PHASES:

        PHASE 1: UPSTATE VALIDATION (Positive Half-Wave)
            When transitioning to positive half-wave:
            1. Validate that a true upstate peak has been reached
            2. Compare actual downstate-to-upstate timing vs predicted timing
            3. Update trough multiplication coefficient based on prediction error
            4. Average recent coefficients to improve future predictions

        PHASE 2: DOWNSTATE DETECTION & PREDICTION (Negative Half-Wave)
            When in the negative half-wave of a slow oscillation:
            1. Extract slow wave onset (positive-to-negative zero crossing)
            2. Update adaptive downstate amplitude threshold
            3. Validate downstate using amplitude and morphology criteria
            4. Calculate actual timestamp of downstate trough (accounting for delays)
            5. Predict upstate timing using adaptive trough multiplication coefficient
            6. Adjust stimulation time to optimize audio cue delivery (2/3 before peak)

        Args:
            uncut_delta (np.ndarray): 30s delta buffer for threshold calculation
            delta (np.ndarray): 5s delta buffer for downstate detection
            slowdelta (np.ndarray): 5s slow delta buffer for upstate validation
            length_threshold (int): Threshold buffer length
            sample_rate (float): EEG sampling rate (Hz)
            current_time (float): Current timestamp (milliseconds)
            cue_duration (float): TMR audio cue length (milliseconds)
            predicted_SO_path (str): File path for logging predictions
        """

        # Extract the signal at first slow oscillation onset sample
        on_delta, idx_p2n = self.extract_slow_oscillation_onset(
            delta)

        if self.is_positive_half_wave and idx_p2n > -1:
            # Retrospective learning about adecuate trough multiplication coefficient
            self.upstate_validation(slowdelta[idx_p2n:])
            # Intuitively, we should use slowDeltaArrayShort(idxP2N:end) here, but it seems to work
            # similarly well with onDelta
            return

        # Adaptive threshold
        self.set_threshold(uncut_delta, current_time, True)

        # Validate slow oscillation downstate
        valid_downstate = self.downstate_validation(on_delta)
        if not valid_downstate:
            return
        
        # Get timestamp of downstate
        downstate_time = self.timestamp_downstate(on_delta,
            current_time, sample_rate)

        # Predict slow oscillations upstate
        stim_at_stamp = self.multiply_through_time(on_delta,
            sample_rate, downstate_time)

        if not self.is_valid_prediction:
            return
        
        self.stim_at_stamp = self.correct_stim_time(stim_at_stamp,
            cue_duration)   # Note that the corrected time can fall
                            # into the past depending on the Cue
                            # sound length

        # We store the upstate time stamp (non-corrected!)
        line = str(downstate_time) + ', Predicted upstate at ' + str(stim_at_stamp)
        print(line)
        self.disk_io.line_store(line, predicted_SO_path)
