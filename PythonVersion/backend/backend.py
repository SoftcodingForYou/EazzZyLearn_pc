from backend.receiver                   import Receiver
from backend.handle_data                import HandleData
from backend.cueing                     import Cueing
from backend.sleep_wake_state           import SleepWakeState
from backend.signal_processing          import SignalProcessing
from backend.predict_slow_oscillation   import PredictSlowOscillation
from threading                          import Thread
from datetime                           import datetime


# Inherance of Receiver class is key here so that we call 
# real_time_alorithm() here below instead of the one in Receiver by 
# fill_buffer()
class Backend(Receiver):

    def __init__(self, gui):
        super().__init__()

        self.print_time_stamp = 0   # Prevent filling console output
        
        self.HndlDt = HandleData('Sleep')
        self.Cng    = Cueing()
        self.Stg    = SleepWakeState()
        self.SgPrc  = SignalProcessing()
        self.Pdct   = PredictSlowOscillation()
        self.gui    = gui

        # Start fading background music
        # background_sound, sampling = self.HndlDt.load_background_sound()
        # self.Cng.start_fading_sound(background_sound, sampling)

        # Start infinite call to real_time_algorithm()
        self.start_receiver(self.HndlDt.output_dir, self.HndlDt.subject_info)
        self.gui.update_status_text("Waiting for data stream ...")


    def real_time_algorithm(self, buffer, timestamps):
        # =================================================================
        # This function is run every time a sample arrives
        # Should be keep as clean as possible and only calls to methods 
        # should be performed
        # =================================================================

        current_time = timestamps[-1] # Used at multiple occasions
        if not self.gui.window_closed:
            dt = datetime.fromtimestamp(current_time/1000) # Needs to be seconds
            formatted_time = dt.strftime("%H:%M:%S")
            self.gui.update_status_text(f"Last samples received: {formatted_time}")

        # Save raw data periodically (periods checked inside method)
        # -----------------------------------------------------------------
        self.HndlDt.master_write_data(buffer, timestamps, self.HndlDt.eeg_path)


        # Extract signals (whole freq range, delta, slow delta)
        # -----------------------------------------------------------------
        v_wake, v_sleep, v_filtered_delta, v_delta, v_slowdelta = self.SgPrc.master_extract_signal(buffer)


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
            self.stop_receiver()
            return
        elif self.gui.stimulation_state != self.softstate:
            self.define_stimulation_state(self.gui.stimulation_state, self.HndlDt.stim_path,
                current_time)
        if self.SgPrc.channel != self.gui.processing_channel - 1:
            self.SgPrc.switch_channel(self.gui.processing_channel, self.HndlDt.stim_path, 
                current_time)
        
        if self.softstate == 0:
            if self.print_time_stamp + 1000 < current_time:
                self.print_time_stamp = current_time
                # print('*** Stimulation paused: Ignoring slow oscillations...')
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
