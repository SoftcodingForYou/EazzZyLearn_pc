import os
import json
from datetime           import datetime
import soundfile        as sf
import numpy            as np
import parameters       as p
from backend.disk_io    import DiskIO


class HandleData():
    """Handles data management for EEG recording sessions, including file preparation,
    sound cue loading, and efficient data storage.

    This class is responsible for:
    - Initializing output directories and files for EEG, stage, stimulation, and prediction data.
    - Formatting and storing subject/session metadata.
    - Loading and managing sound cues and background sounds for stimulation.
    - Efficiently buffering and writing EEG data to disk using background threads.
    - Providing utility methods for directory and file management, as well as sound file handling.

    Attributes:
        encoding (str): File encoding used for output files.
        sample_rate (int): Sampling rate for EEG data.
        cue_dir (str): Directory containing sound cue files.
        chosen_cue (str): Filename of the chosen sound cue.
        duration (float): Duration of the chosen sound cue in milliseconds.
        soundsampling (int): Sampling rate of the chosen sound cue.
        soundarray (np.ndarray): Loaded sound cue as a NumPy array.
        subject_info (dict): Metadata about the subject.
        output_dir (str): Directory for storing session output files.
        eeg_path (str): Path to the EEG data file.
        stage_path (str): Path to the sleep/wake stage data file.
        stim_path (str): Path to the stimulation data file.
        pred_path (str): Path to the prediction data file.
        sample_count (int): Counter for buffered EEG samples.
        saving_interval (int): Interval for saving data to disk.
        samples_in_buffer (int): Number of samples to buffer before saving.
        background_sound (str): Filename of the background sound.
        sound_format (str): File format for sound cues.
        disk_io (DiskIO): DiskIO instance for background file writing.
    """

    def __init__(self, session):
        """Initializes output-related attributes and prepares disk storage.

        - Stores subject information in self.subject_info.
        - Creates the output directory in self.output_dir.
        - Initializes files with headers containing recording information:
            * EEG data: self.eeg_path
            * Sleep and wake stage information: self.stage_path
            * Stimulation information: self.stim_path
        - Loads sound cues into memory as numpy arrays, including:
            * Sound durations (ms)
            * Sampling rates
        """

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
        """Ensures the output directory for the subject exists, creating it if necessary.

        - Constructs the full path for the subject's output directory.
        - Creates the directory (and any necessary parent directories) if it does not already exist.
        - Returns the full path to the subject's output directory.

        Args:
            output_dir (str): The base directory where subject data should be stored.
            subject_info (dict): Dictionary containing subject information, must include 'name'.

        Returns:
            str: The full path to the subject's output directory.
        """

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
        """Format subject name to delete spaces and replace them with underscores

        Args:
            subject_name (str): Name of output files

        Returns:
            subject_name (str): Formatted name without spaces
        """

        subject_name = subject_name.strip() # Delete spaces at the start and end
        subject_name = subject_name.replace(' ', '_') # Replace inter spaces with underscores
        return subject_name


    def prep_files(self, output_dir, subject_info, elecs, idx_elec, 
        default_threshold, artifact_threshold, length_refractory, appendix):
        """Prepares and initializes a data file for recording session information.

        - Constructs a unique file path based on the subject's name and the current date/time.
        - Creates the file and writes a header line containing session and subject metadata in JSON format.
        - Returns the path to the created file for later data appending.

        Args:
            output_dir (str): Base directory for output files.
            subject_info (dict): Dictionary with subject metadata (name, age, sex, chosencue).
            elecs (list): List of electrode names or identifiers.
            idx_elec (int or str): Index or identifier of the electrode used for stimulation/recording.
            default_threshold (float): Default threshold value for event detection.
            artifact_threshold (float): Threshold for artifact rejection.
            length_refractory (float): Refractory period duration.
            appendix (str): Suffix to append to the filename (e.g., "_eeg.txt").

        Returns:
            str: Full path to the initialized data file.
        """

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
        """Periodically saves buffered EEG data and timestamps to disk.

        This method increments an internal sample counter each time it is called.
        When the number of calls reaches the buffer size (`self.samples_in_buffer`),
        it resets the counter and writes the accumulated EEG data and corresponding
        timestamps to the specified output file using a background thread for efficiency.

        The data is formatted as CSV lines, with each row containing a timestamp
        followed by the EEG data for that sample.

        Args:
            eeg_data (np.ndarray): 2D array of EEG data (channels x samples).
            time_stamps (np.ndarray): 1D array of timestamps corresponding to each sample.
            output_file (str): Path to the file where the data should be appended.
        """

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
        """Dynamically determines the directory path for sound cue files.

        This method constructs the absolute path to the 'Sounds' directory
        based on the current file's location, ensuring compatibility across
        different platforms and avoiding hard-coded paths. It searches for
        the base project directory ('EazzZyLearn_pc') in the current file's
        path and appends '/Sounds/' to it.

        Returns:
            str: The absolute path to the sound cues directory.
        """

        dirname       = os.path.dirname(__file__)
        str2find      = 'EazzZyLearn_pc'
        idx_base      = dirname.find(str2find)
        base_path     = dirname[0:idx_base+len(str2find)]
        cue_directory = base_path + r'/Sounds/'

        return cue_directory


    def prep_cue_info(self, cue_dir, cue):
        """Retrieves information about a sound cue file.

        This method opens the specified sound file and extracts its sampling rate and duration in 
        milliseconds. It is used to gather metadata about the cue for later use, such as playback 
        timing and compatibility checks.

        Args:
            cue_dir (str): The directory containing the sound cue files.
            cue (str): The filename of the cue to analyze.

        Returns:
            tuple: (duration, samplingrate)
                duration (float): Duration of the sound cue in milliseconds.
                samplingrate (int): Sampling rate of the sound cue in Hz.
        """
        
        current_sound       = sf.SoundFile(cue_dir + cue)
        samplingrate        = current_sound.samplerate
        duration            = (current_sound.frames / 
                                current_sound.samplerate) * 1000

        return duration, samplingrate


    def prep_cue_load(self, cue_dir, cue):
        """Loads a sound cue file into memory as a NumPy array.

        This method reads the specified sound file from disk and stores it as a NumPy array,
        which helps reduce stimulation delay by avoiding repeated disk access.

        Args:
            cue_dir (str): Directory containing the sound cue files.
            cue (str): Filename of the cue to load.

        Returns:
            np.ndarray: The loaded sound cue as a NumPy array.
        """

        soundarray, fs = sf.read(cue_dir + cue, dtype='float32')
        
        return soundarray