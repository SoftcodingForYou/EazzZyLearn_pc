from backend.receiver                   import Receiver
from backend.handle_data                import HandleData
from backend.cueing                     import Cueing
from backend.sleep_wake_state           import SleepWakeState
from backend.signal_processing          import SignalProcessing
from backend.predict_slow_oscillation   import PredictSlowOscillation
from threading                          import Thread
from datetime                           import datetime
import time

# Inherance of Receiver class is key here so that we call 
# real_time_alorithm() here below instead of the one in Receiver by 
# fill_buffer()
class Backend(Receiver):
    """
    Main backend controller for EazzZyLearn real-time processing.

    Features:
    - Inherits from Receiver to handle real-time EEG data acquisition.
    - Coordinates data handling, cueing, sleep/wake staging, signal processing, and slow oscillation prediction.
    - Interfaces with the GUI to update status and respond to user actions (enable, force, pause, channel selection).
    - Runs a runtime monitor thread to report processing speed and sample reception.
    - Implements the real_time_algorithm for per-sample processing, including:
        - Data saving
        - Signal extraction
        - Sleep/wake staging
        - Stimulation state management
        - Slow oscillation upstate prediction
        - Stimulation triggering

    Args:
        gui: Reference to the Frontend instance for UI updates and state checks.

    Usage:
        backend = Backend(gui)
    """

    def __init__(self, gui):
        super().__init__()

        self.print_time_stamp = 0   # Prevent filling console output
        
        self.HndlDt = HandleData('Sleep')
        self.Cng    = Cueing()
        self.Stg    = SleepWakeState()
        self.SgPrc  = SignalProcessing()
        self.Pdct   = PredictSlowOscillation()
        self.gui    = gui
        gui.backend = self

        # Start fading background music
        # background_sound, sampling = self.HndlDt.load_background_sound()
        # self.Cng.start_fading_sound(background_sound, sampling)

        # Start infinite call to real_time_algorithm()
        self.start_receiver(self.HndlDt.output_dir, self.HndlDt.subject_info)
        self.gui.update_status_text("Waiting for data stream ...")

        self.current_time       = float(0.0)
        self.monitor_iterations = int(0)
        self.monitor_interval   = int(5) # seconds
        self.monitor_running    = True
        
        # Buffer copy for monitoring (updated periodically)
        self.monitor_buffer     = None # Whole range passband filtered signal
        self.monitor_buffer2    = None # Delta filtered signal
        self.monitor_timestamps = None
        
        self.monitor_thread     = Thread(
            target=self.runtime_monitor, daemon=True, name='runtime_monitor')
        self.monitor_thread.start()

        # Sound feedback loop state
        self.debugging_sound_loop_thread = None


    def real_time_algorithm(self, buffer, timestamps):
        # =================================================================
        # This function is run every time a sample arrives
        # Should be keep as clean as possible and only calls to methods 
        # should be performed
        # =================================================================

        current_time = timestamps[-1] # Used at multiple occasions
        self.current_time = current_time
        self.monitor_iterations += 1


        # Save raw data periodically (periods checked inside method)
        # -----------------------------------------------------------------
        self.HndlDt.master_write_data(buffer, timestamps, self.HndlDt.eeg_path)


        # Debugging option: Check for sound-EEG feedback loop mode
        # -----------------------------------------------------------------
        if self.gui.sound_feedback_enabled:
            # Check if thread is None or not alive (finished playing)
            if self.debugging_sound_loop_thread is None or not self.debugging_sound_loop_thread.is_alive():
                self.debugging_sound_loop_thread = Thread(target=self.Cng.master_cue_stimulate,
                    args=(self.HndlDt.chosen_cue, self.HndlDt.soundarray,
                    self.HndlDt.soundsampling, self.HndlDt.cue_dir,
                    self.HndlDt.stim_path, current_time))
                self.debugging_sound_loop_thread.daemon = True
                self.debugging_sound_loop_thread.start()
            return  # Skip all other processing


        # Extract signals (whole freq range, delta, slow delta)
        # -----------------------------------------------------------------
        v_wake, v_sleep, v_filtered_delta, v_delta, v_slowdelta = self.SgPrc.master_extract_signal(buffer)


        # Update monitor buffer periodically (only if plotting is enabled)
        if self.gui.plot_enabled and self.monitor_iterations == self.HndlDt.sample_rate * self.monitor_interval:
            self.monitor_buffer = v_sleep.copy()
            self.monitor_buffer2 = v_filtered_delta.copy()
            self.monitor_timestamps = timestamps.copy()


        # Sleep and wake staging
        # -----------------------------------------------------------------
        if self.use_muse_sleep_classifier:
            self.Stg.current_muse_metrics = self.current_muse_metrics
        self.Stg.master_stage_wake_and_sleep(v_wake, v_sleep,
            self.SgPrc.freq_range_whole, self.HndlDt.stage_path,
            current_time)


        # Checkpoint of whether to proceed to stimulation or to step out
        # -----------------------------------------------------------------
        # At this stage, we allow for code to be soft-paused or -forced: We
        # block or force stimulation manually
        if self.gui.window_closed and not self.stop:
            self.monitor_running = False
            self.stop_receiver()
            return
        elif self.gui.stimulation_state != self.softstate:
            self.define_stimulation_state(self.gui.stimulation_state)
        if self.SgPrc.channel != self.gui.processing_channel - 1:
            self.SgPrc.switch_channel(self.gui.processing_channel, self.HndlDt.stim_path, 
                current_time)
        
        if self.softstate == 0:
            # if self.print_time_stamp + 1000 < current_time:
            #     self.print_time_stamp = current_time
            #     print('*** Stimulation paused: Ignoring slow oscillations...')
            return
        elif self.softstate != -1 and ( 
            self.Stg.isawake == True or self.Stg.issws == False ):
            return # Ignored if we want to force stimulations
        elif self.softstate == -1 and ( self.print_time_stamp + 
            1000 < current_time ):
                self.print_time_stamp = current_time
                # print('*** Stimulation forced: Ignoring stagings...')


        # Slow oscillation upstate prediction
        # -----------------------------------------------------------------
        if self.Pdct.stim_at_stamp == None:
            self.Pdct.master_slow_osc_prediction(v_filtered_delta,
                v_delta, v_slowdelta, self.SgPrc.threshold_buffer_length,
                self.HndlDt.sample_rate, current_time,
                self.HndlDt.duration, self.HndlDt.pred_path)


        # Stimulate
        # -----------------------------------------------------------------
        # Stimulation triggering if the following conditions are respected:
        # 1. there is a valid stimulation time
        # 2. the determined stimulation time matches the current cpu time
        #    or CPU time already ahead
        # 3. We are not inside a refractory window following a previous 
        #    stimulation
        if ( self.Pdct.stim_at_stamp != None and 
            current_time >= self.Pdct.stim_at_stamp and
            current_time <= self.Cng.stim_time + self.Cng.len_refractory ):
            
            self.Pdct.stim_at_stamp = None  # Important to reset if 
                                            # supposed stim fell inside 
                                            # refractory period 
        
        elif ( self.Pdct.stim_at_stamp != None and 
            current_time >= self.Pdct.stim_at_stamp ):

            # Set important values OUTSIDE of thread in order to update quickly
            self.Cng.stim_time      = current_time # Crucial for refractory period
            self.Pdct.stim_at_stamp = None # Crucial to reset
            stim_thread             = Thread(target=self.Cng.master_cue_stimulate, 
                args=(self.HndlDt.chosen_cue, self.HndlDt.soundarray,
                self.HndlDt.soundsampling, self.HndlDt.cue_dir,
                self.HndlDt.stim_path, current_time))
            stim_thread.start()

        # print(self.get_time_stamp() - current_time) # Evaluate code speed


    def runtime_monitor(self):
        """Separate thread for monitoring and reporting runtime monitor"""
        while self.monitor_running:
            time.sleep(self.monitor_interval)
            if self.monitor_iterations > 0:
                monitor = round(self.monitor_iterations / self.monitor_interval)
                self.gui.update_speed_text(f"Runtime speed: {monitor} Hz")
                self.monitor_iterations = 0

            # Update sleep states display
            self.gui.update_sleep_states(self.Stg.isawake, self.Stg.issws)

            # Update plot if buffer is available and plotting is enabled
            if self.gui.plot_enabled and self.monitor_buffer is not None:
                # Safe to access monitor_buffer - it's a copy updated periodically
                # Pass both buffers to GUI for plotting
                self.gui.update_plot(self.monitor_buffer, self.monitor_buffer2)

            dt = datetime.fromtimestamp(self.current_time/1000) # Needs to be seconds
            formatted_time = dt.strftime("%H:%M:%S")
            self.gui.update_status_text(f"Last samples received: {formatted_time}")