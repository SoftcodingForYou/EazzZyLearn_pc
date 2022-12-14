'''
This script runs OpenBCI formatted files without the need to run them through 
the actual OpenBCI GUI application.

                            <<< IMPORTANT >>>
                            =================

Depending on whether this script is run on signals with correct or 
incorrect polarity, the parameter to flip the signs can be 
enabled/disabled. During the code iteration the signals will ***NOT*** be 
flipped because the sample fetching step is ommitted in this simulation!

The parameter subject name in parameters.py will be overwritten according 
to the current subject that is run in the for loop.
'''


# Parameters --------------------------------------------------------------
pathdata                = 'C:\\Users\\david\\OneDrive\\Documents\\Helment\\SampleDataOpenBCIFormat'

signal_sign_flipping    = 1 # [0, 1] for yes or no


# Prepare userland --------------------------------------------------------
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from threading import Thread


# Locate and import real-time functions
from backend.handle_data import HandleData
from backend.cueing import Cueing
from backend.predict_slow_oscillation import PredictSlowOscillation
from backend.signal_processing import SignalProcessing
from backend.sleep_wake_state import SleepWakeState
import parameters as p


# Locate subjects to run
files                   = os.listdir(pathdata)
files                   = [file for file in files if '.ini' not in file]
files.sort() # Important

# Magic zippidy dee for loop just "for" you -------------------------------
for file in files:
    
    print('<!> Signal polarity conversion state is set to: ' + str(signal_sign_flipping))

    print('Reading and formatting data for ' + file)

    output_name = file.replace('.txt', '')
    p.SUBJECT_INFO["name"] = output_name

    data                = open(pathdata + '/' + file, 'r')
    data                = data.read()
    data_list           = data.split('\n')

    # Define header information
    iLine               = 0
    for line in data_list:
        if '%Sample Rate = ' in line:
            start       = line.find('= ')
            stop        = line.find(' Hz')
            sampling_rate = int(line[start+2:stop])
        if line[0] != '%':
            data_start  = iLine + 1
            break
        iLine           = iLine + 1

    columns             = data_list[iLine].split(',')
    for iCol in range(len(columns)):
        columns[iCol]   = columns[iCol].replace(' ', '')
    
    # For some reason there is an empty element in the beginning of the 
    # list that is not appearing later when going through the samples
    columns = columns[1:]

    # Locate samples, time stamps and electrodes inside data
    idx_samples         = [iCol for iCol in range(len(columns))
                            if 'SamplingIndex' in columns[iCol]]
    
    idx_timestamps = [idx for idx in range(len(columns)) if columns[idx] == 'Timestamp']

    idx_electrodes = []
    for elec in list(p.ELEC.keys()):
        for iCol in range(len(columns)):
            if columns[iCol] == elec:
                idx_electrodes.append(iCol)


    data_list           = data_list[data_start:]


    print('Populating EEG data matrix. This is inefficiently coded and will take a while ...')
    eeg_data            = np.zeros((len(idx_electrodes), len(data_list)))
    times               = np.zeros((len(data_list)))
    for iSample in range(len(data_list)):
        
        line            = data_list[iSample] # Extract row
        line            = line.replace(" ", "") # Homogeination

        if line == '': # Last line
            continue

        line_list       = line.split(',')
        line_list       = line_list[0:-1] # Last one is not convertable into float

        data_points     = np.asarray(list(map(float, line_list)))

        times[iSample]      = float(data_points[idx_timestamps])
        eeg_data[:,iSample] = data_points[idx_electrodes]


    if signal_sign_flipping == 1:
        eeg_data = eeg_data * -1
        print('... I just flipped the signal signs')


    # We now have a time stamp vector and a [channel x samples] matrix as 
    # we would from OpenBCI GUI into the real-time detection code

    # Simulate real-time recording ----------------------------------------        
    HndlDt              = HandleData()
    Cng                 = Cueing()
    Stg                 = SleepWakeState()
    SgPrc               = SignalProcessing()
    Pdct                = PredictSlowOscillation()


    def real_time_algorithm(buffer, timestamps):

        current_time = timestamps[-1] # Used at multiple occasions

        # Save raw data periodically (periods checked inside method)
        # -----------------------------------------------------------------
        HndlDt.master_write_data(buffer, timestamps, HndlDt.eeg_path)


        # Extract signals (whole freq range, delta, slow delta)
        # -----------------------------------------------------------------
        v_wake, v_sleep, v_filtered_delta, v_delta, v_slowdelta = SgPrc.master_extract_signal(buffer)


        # Sleep and wake staging
        # -----------------------------------------------------------------
        Stg.master_stage_wake_and_sleep(v_wake, v_sleep,
            SgPrc.freq_range_whole, HndlDt.stage_path,
            current_time)


        if Stg.isawake == True or Stg.issws != True:
            return


        # Slow oscillation upstate prediction
        # -----------------------------------------------------------------
        if Pdct.stim_at_stamp == None:
            Pdct.master_slow_osc_prediction(v_filtered_delta,
                v_delta, v_slowdelta, SgPrc.threshold_buffer_length,
                HndlDt.sample_rate, current_time, HndlDt.duration, 
                HndlDt.pred_path)


        # Stimulate
        # -----------------------------------------------------------------
        # Stimulation triggering if the following conditions are respected:
        # 1. there is a valid stimulation time
        # 2. the determined stimulation time matches the current cpu time
        #    or CPU time already ahead
        # 3. We are not inside a refractory window following a previous 
        #    stimulation
        if ( Pdct.stim_at_stamp != None and 
            current_time >= Pdct.stim_at_stamp and
            current_time <= Cng.stim_time + Cng.len_refractory ):
            
            Pdct.stim_at_stamp = None  # Important to reset if 
                                        # supposed stim fell inside 
                                        # refractory period 
        
        elif ( Pdct.stim_at_stamp != None and 
            current_time >= Pdct.stim_at_stamp ):

            # Set important values OUTSIDE of thread in order to update quickly
            Cng.stim_time      = current_time # Crucial for refractory period
            Pdct.stim_at_stamp = None # Crucial to reset
            stim_thread             = Thread(target=Cng.master_cue_stimulate, 
                args=(HndlDt.chosen_cue, HndlDt.soundarray,
                HndlDt.soundsampling, HndlDt.cue_dir,
                HndlDt.stim_path, current_time))
            stim_thread.start()

    
    time_stamps         = np.zeros(p.MAIN_BUFFER_LENGTH)
    buffer              = np.zeros((eeg_data.shape[0], p.MAIN_BUFFER_LENGTH))
    for iSample in range(len(times)):

        sample          = eeg_data[:, iSample]
        sample          = np.reshape(sample, (-1, 1))
        update_buffer   = np.concatenate((buffer, sample), axis=1)
        buffer          = update_buffer[:, 1:]

        time_stamps     = np.append(time_stamps, times[iSample])
        time_stamps     = time_stamps[1:]

        real_time_algorithm(buffer, time_stamps)


print('Script finished normally')
        
        