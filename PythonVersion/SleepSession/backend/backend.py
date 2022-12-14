from backend.receiver import Receiver
from backend.handle_data import HandleData
from backend.cueing import Cueing
from backend.sleep_wake_state import SleepWakeState
from backend.signal_processing import SignalProcessing
from backend.predict_slow_oscillation import PredictSlowOscillation
import keyboard
from threading import Thread
import numpy as np


# Inherance of Receiver class is key here so that we call 
# real_time_alorithm() here below instead of the one in Receiver by 
# fill_buffer()
class Backend(Receiver):

    def __init__(self):
        super().__init__()

        self.print_time_stamp = 0   # Prevent filling console output
        
        self.HndlDt = HandleData('Sleep')
        self.Cng    = Cueing()
        self.Stg    = SleepWakeState()
        self.SgPrc  = SignalProcessing()
        self.Pdct   = PredictSlowOscillation()

        # Start infinite call to real_time_algorithm()
        self.start_receiver(self.HndlDt.output_dir, self.HndlDt.subject_info)


    def real_time_algorithm(self, buffer, timestamps):
        # =================================================================
        # This function is run every time a sample arrives
        # Should be keep as clean as possible and only calls to methods 
        # should be performed
        # =================================================================

        current_time = timestamps[-1] # Used at multiple occasions

        # Save raw data periodically (periods checked inside method)
        # -----------------------------------------------------------------
        self.HndlDt.master_write_data(buffer, timestamps, self.HndlDt.eeg_path)


        # Extract signals (whole freq range, delta, slow delta)
        # -----------------------------------------------------------------
        v_wake, v_sleep, v_filtered_delta, v_delta, v_slowdelta = self.SgPrc.master_extract_signal(buffer)


        # Sleep and wake staging
        # -----------------------------------------------------------------
        self.Stg.master_stage_wake_and_sleep(v_wake, v_sleep,
            self.SgPrc.freq_range_whole, self.HndlDt.stage_path,
            current_time)


        # Checkpoint of whether to proceed to stimulation or to step out
        # -----------------------------------------------------------------
        # Soft interrupt (Comment out on GNU/Linux because of sudo request)
        # At this stage, we allow for code to be soft-paused or -forced: We
        # block or force stimulation manually
        if any(keyboard.is_pressed(key) for key in ["p", "r", "f", "q"]):
            key = keyboard.read_key()
            self.define_stimulation_state(key, self.HndlDt.stim_path,
                current_time)
        elif any(keyboard.is_pressed(key) for key in map(str, np.arange(1,9))):
            number = keyboard.read_key()
            self.SgPrc.switch_channel(number, self.HndlDt.stim_path, current_time)
        
        if self.softstate == 'paused':
            if self.print_time_stamp + 1000 < current_time:
                self.print_time_stamp = current_time
                print('*** Stimulation paused: Ignoring slow oscillations...')
            return
        elif self.softstate != 'forced' and ( 
            self.Stg.isawake == True or self.Stg.issws != True ):
            return # Ignored if we want to force stimulations
        elif self.softstate == 'forced' and ( self.print_time_stamp + 
            1000 < current_time ):
                self.print_time_stamp = current_time
                print('*** Stimulation forced: Ignoring stagings...')


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

        



if __name__ == '__main__':
    import os; os.system('clear')
    backend = Backend()
    
    # Subject info
    subject_info = {
        'name': 'Carlos',
        'age': 26,
        'sex': 'Male',
        'head_measure': 59,
        'sample_rate': 250
    }
    output_dir = ''  # current folder

    backend.start(output_dir, subject_info)
