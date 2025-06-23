import os
import json
from datetime           import datetime
import soundfile        as sf
import numpy            as np
import parameters       as p
from backend.disk_io    import DiskIO


class HandleData():

    def __init__(self, session):
        # =================================================================
        # Initialize outputs on disk
        # -----------------------------------------------------------------
        # - Stores subject information: self.subject_info
        # - Create output directory: self.output_dir
        # - Initialize files with headers containing recording information
        #    * eeg data: self.eeg_path
        #    * sleep and wake stage information: self.stage_path
        #    * Stimulation information: self.stim_path
        # - Get information about sound cues and load them into memory
        #    * Sounds as numpy arrays
        #    * Sound durations (ms)
        #    * Sampling rates
        # =================================================================

        self.encoding       = p.ENCODING
        self.sample_rate    = p.SAMPLERATE

        # create subject folder is not present
        # only if output_dir and subject_info are present
        if p.OUTPUT_DIR is None or p.SUBJECT_INFO is None:
            raise Exception('Define output and subject information')

        # Prepare sound cues
        # -----------------------------------------------------------------
        # Defines directory containing sounds
        self.cue_dir     = self.prep_cue_dir()
        self.chosen_cue  = p.SUBJECT_INFO["chosencue"] + p.SOUND_FORMAT
        # Get information about cues (sampling freq, duration)
        self.duration, self.soundsampling = self.prep_cue_info(
            self.cue_dir, self.chosen_cue)
        # Preallocte cues into memory
        self.soundarray = self.prep_cue_load(self.cue_dir, self.chosen_cue)

        # Prepare output directory and files
        # -----------------------------------------------------------------
        self.subject_info = p.SUBJECT_INFO

        if session == 'Sleep':
            self.output_dir = self.prep_output_dir(p.OUTPUT_DIR, p.SUBJECT_INFO)
            self.eeg_path   = self.prep_files(p.OUTPUT_DIR, p.SUBJECT_INFO, 
                p.ELEC, p.IDX_ELEC, p.DEFAULT_THRESHOLD, 
                p.NON_PHYSIOLOGICAL_THRESHOLD, p.LEN_REFRACTORY, "_eeg.txt")
            self.stage_path = self.prep_files(p.OUTPUT_DIR, p.SUBJECT_INFO, 
                p.ELEC, p.IDX_ELEC, p.DEFAULT_THRESHOLD, 
                p.NON_PHYSIOLOGICAL_THRESHOLD, p.LEN_REFRACTORY, "_stage.txt")
            self.stim_path  = self.prep_files(p.OUTPUT_DIR, p.SUBJECT_INFO, 
                p.ELEC, p.IDX_ELEC, p.DEFAULT_THRESHOLD, 
                p.NON_PHYSIOLOGICAL_THRESHOLD, p.LEN_REFRACTORY, "_stim.txt")
            self.pred_path  = self.prep_files(p.OUTPUT_DIR, p.SUBJECT_INFO, 
                p.ELEC, p.IDX_ELEC, p.DEFAULT_THRESHOLD, 
                p.NON_PHYSIOLOGICAL_THRESHOLD, p.LEN_REFRACTORY, "_pred.txt")

        # Parameters for saving of EEG data
        # -----------------------------------------------------------------
        self.sample_count       = 0
        self.saving_interval    = p.DATA_SAVE_INTERVAL
        self.samples_in_buffer  = p.MAIN_BUFFER_LENGTH

        # Background sound
        self.background_sound   = p.SUBJECT_INFO["background"]
        self.sound_format       = p.SOUND_FORMAT

        self.disk_io            = DiskIO(
            p.MAIN_BUFFER_LENGTH, p.DATA_FLUSH_INTERVAL, 'handle_data_thread')


    def load_background_sound(self):
        return self.prep_cue_load(
            self.cue_dir, self.background_sound + self.sound_format
            ), self.soundsampling
        

    def prep_output_dir(self, output_dir, subject_info):
        # Check that a folder exists, otherwise creates one

        # Format subject name
        subject_name = self.format_subject_name(subject_info['name'])

        # Full path of the folder to be saved
        full_path = os.path.join(output_dir, subject_name) 

        # Si la carpeta no existe crea una
        if not os.path.isdir(full_path):
            # Allow for recursive folder generation with os.makedirs
            os.makedirs(full_path, exist_ok = True)

        return full_path


    def format_subject_name(self, subject_name):
        # Format subject name to delete spaces and replace them with underscores
        subject_name = subject_name.strip() # Delete spaces at the start and end
        subject_name = subject_name.replace(' ', '_') # Replace inter spaces with underscores
        return subject_name


    def prep_files(self, output_dir, subject_info, elecs, idx_elec, 
        default_threshold, artifact_threshold, length_refractory, appendix):
        # Subject info keys and values
        subject_name    = subject_info['name']
        subject_age     = subject_info['age']
        subject_sex     = subject_info['sex']
        chosencue       = subject_info["chosencue"]

        # Format subject name
        subject_name = self.format_subject_name(subject_info['name'])

        # name structure
        date = datetime.now()
        date = date.strftime("%d-%m-%Y_%H-%M-%S")
        
        # Full path to save the data
        full_path = os.path.join(output_dir, subject_name, date + "_" + subject_name )

        # Save directions to write to them later
        data_path = full_path + appendix

        # create eeg_data file
        with open(data_path, 'w', encoding=self.encoding) as file:
            line = {
                'date':                 date, 
                'sample freq':          str(self.sample_rate), 
                'subject name':         subject_name, 
                'age':                  subject_age, 
                'sex':                  subject_sex, 
                'chosencue':            chosencue, 
                'electrodes':           json.dumps(elecs),
                'used elec':            idx_elec,
                'default threshold':    default_threshold,
                'artifact threshold':   artifact_threshold,
                'refractory duration':  length_refractory}
            file.write(json.dumps(line) + "\n\n\n")
            file.close() # Important for data to get written

        return data_path
    

    def master_write_data(self, eeg_data, time_stamps, output_file):
        # =================================================================
        # This method verifies if the EEG data has to be saved since saving
        # occurs perioically. If conditions are satisfied, it calls for a 
        # thread that will save the EEG data chunk on disk
        # =================================================================
        self.sample_count   = self.sample_count + 1
        if self.sample_count < self.samples_in_buffer:
            return # too early
        self.sample_count   = 0 # Important: reset outside of thread
        
        # Use numpy's savetxt for maximum performance
        # Reshape data to combine timestamps and EEG data
        combined_data = np.column_stack([time_stamps, eeg_data.T])
        data_string = '\n'.join([', '.join(map(str, row)) for row in combined_data])

        self.disk_io.line_store(data_string, output_file)


    def prep_cue_dir(self):
        # =================================================================
        # Small method defining the sound cue directory dynamically to
        # avoid hard-coding when changing platforms
        # =================================================================
        dirname       = os.path.dirname(__file__)
        str2find      = 'EazzZyLearn_pc'
        idx_base      = dirname.find(str2find)
        base_path     = dirname[0:idx_base+len(str2find)]
        cue_directory = base_path + r'/Sounds/'

        return cue_directory


    def prep_cue_info(self, cue_dir, cue):
        # =================================================================
        # Get several information about stimulation cue files and store the
        # cues in self so that files don't need to be read on the fly: Save
        # processing time
        # =================================================================
        current_sound       = sf.SoundFile(cue_dir + cue)
        samplingrate        = current_sound.samplerate
        duration            = (current_sound.frames / 
                                current_sound.samplerate) * 1000

        return duration, samplingrate


    def prep_cue_load(self, cue_dir, cue):
        # =================================================================
        # This method stores sound cues as arrays in memory. This is 
        # crucial in order to to reduce the stimulation delay after calling 
        # for stimulation compared to loading the sound file from disk on  
        # the fly.
        # ================================================================= 
        soundarray, fs = sf.read(cue_dir + cue, dtype='float32')
        
        return soundarray