# Collection of methods associated with memory cueing.
import sounddevice      as sd
import numpy            as np
sd.default.latency = 'low' # Crucial to reduce onset latency of cues
import parameters       as p
from backend.disk_io    import DiskIO


class Cueing:

    def __init__(self):
        # =================================================================
        # Initialize cues
        # -----------------------------------------------------------------
        # - Initialize time variables that need to be met before 
        #   stimulation
        # - Randomized order of stimulation of cues
        # =================================================================
        
        # Post-stimulation resting period for brain
        self.studycueinterval   = float(p.SUBJECT_INFO["cueinterval"]) * 60000 # from min to ms
        self.len_refractory     = p.LEN_REFRACTORY * 1000
        self.stim_time          = 0 # Last stimulation time stamp (Used to respect refractory periods)
        self.s_stim             = 0 # Keep count of stimulations (scalar)

        self.disk_io            = DiskIO(p.MAX_BUFFERED_LINES, p.STIM_FLUSH_INTERVAL)


    def start_fading_sound(self, sound, soundsampling):
        # Fade sound array
        sound_faded = sound * np.arange(1, 0, -1/sound.size) # Slowly decreasing amplitude to 0
        # Launch background ambient sound
        sd.play(sound_faded, soundsampling, blocking = False, loop = False)

    
    def cue_play(self, cue_array, sample_rate):
        # =================================================================
        # Simple methid that plays back a numpy array as sound.
        # playsound library is not recommended as it has high latency.
        # =================================================================
        sd.play(cue_array, sample_rate)
        # sd.wait() # Stops python interpreter until sound played entirely


    def master_cue_stimulate(self, cue, cue_array, cue_sampling_rate, 
        cue_dir, output_file, current_time):
        # =================================================================
        # This method simply plays back the sound.
        # =================================================================  

        # Play back cue
        self.cue_play(cue_array, cue_sampling_rate)

        # Store the stimulation time and cue
        line = str(current_time) + ', ' + cue + ' ' + cue_dir
        self.disk_io.line_store(line, output_file)

        # Post-stimulation cleanup
        # -----------------------------------------------------------------
        self.s_stim        = self.s_stim + 1