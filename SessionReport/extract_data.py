import os
import numpy as np
import mne
from datetime import datetime, timezone
import json
import scipy


class DataExtraction():
    
    def master_extraction(self, pathdata):

        files = os.listdir(pathdata)
        files = [f for f in files if f[0:2] != '._']  # System trash
        files.sort()

        # List output types
        eeg_files = []  # Contains RAW EEG signal
        pred_files = []  # Contains time stamps of predicted upstates
        stim_files = []  # Contains information about stimulations
        stage_files = []  # Contains sleep/wake stages
        for idx_file in range(len(files)):
            if '_eeg.txt' in files[idx_file]:
                eeg_files.append(files[idx_file])
            elif '_pred.txt' in files[idx_file]:
                pred_files.append(files[idx_file])
            elif '_stim.txt' in files[idx_file]:
                stim_files.append(files[idx_file])
            elif '_stage.txt' in files[idx_file]:
                stage_files.append(files[idx_file])

        # Of crucial importance!! os.listdir generates a list OF RANDOM ORDER
        # that moreover can change between operating systems!
        eeg_files.sort()
        stage_files.sort()
        pred_files.sort()
        stim_files.sort()

        # Extract header
        # -----------------------------------------------------------------
        with open(os.path.join(pathdata, eeg_files[0]), 'r', encoding='utf8') as file:
            lines = file.readlines()
        eeg_info = lines[0]  # Header information of the eeg files
        eeg_info = self.extract_hdr_info(eeg_info)
        s_fs = int(eeg_info["sample freq"])

        # Extract EEG signals (Pool the information in the different files)
        # -----------------------------------------------------------------
        # New files have start time as zero. We correct times relative to previous files
        eeg_lat_shift = 0
        eeg_out = []
        pred_out = []
        stim_out = []
        stage_out = []
        t_file_breaks = []

        for iFile in range(len(eeg_files)):

            # Visual inspection of data consistency
            print('Loading ' + eeg_files[iFile])
            # print('Loading ' + pred_files[iFile])
            # print('Loading ' + stim_files[iFile])
            # print('Loading ' + stage_files[iFile])

            eeg_curr, last_timestamp = self.extract_data(
                pathdata, eeg_files[iFile], eeg_lat_shift, 'eeg')
            pred_curr, _ = self.extract_data(
                pathdata, pred_files[iFile], eeg_lat_shift, 'pred')
            stim_curr, _ = self.extract_data(
                pathdata, stim_files[iFile], eeg_lat_shift, 'stim')
            stage_curr, _ = self.extract_data(
                pathdata, stage_files[iFile], eeg_lat_shift, 'stage')

            # Pool
            eeg_out = eeg_out + eeg_curr
            pred_out = pred_out + pred_curr
            stim_out = stim_out + stim_curr
            stage_out = stage_out + stage_curr

            eeg_lat_shift = last_timestamp
            t_file_breaks.append(last_timestamp)

        # Build EEG signal and time stamps
        # -----------------------------------------------------------------
        number_elecs = len(eeg_info['electrodes'].values())
        eeg_data = np.zeros((number_elecs, len(eeg_out)))
        times = np.zeros((len(eeg_out)))
        idx_reject = []
        for iSample in range(len(eeg_out)):

            line = eeg_out[iSample]  # Extract row
            line = line.replace(" ", "")  # Homogeination

            data_list = line.split(',')
            if len(data_list) < number_elecs+1:
                print('Skipping incomplete EEG line')
                idx_reject.append(iSample)
                continue
            elif 'nan' in data_list:
                idx_reject.append(iSample)
                continue
            data_points = np.asarray(list(map(float, data_list)))

            times[iSample] = data_points[0]
            eeg_data[:, iSample] = data_points[1:]

        # Reject NaNs
        times = np.delete(times, idx_reject)
        eeg_data = np.delete(eeg_data, idx_reject, 1)

        # Important: "times" are spaced around 1.5 ms between samples. This
        # is the speed of the algorithm. Every 10 samples, there is an
        # increased interval to up to 20ms. This is because of the
        # buffering by the Cyton board. The GUI receives a message of 10
        # samples and forwards sample-by-sample to our algorithm.

        # Extract monitored signal (including channel switches)
        channels = list(eeg_info["electrodes"].keys())
        default_channel = channels[eeg_info["used elec"]]
        eeg_monitor, t_switches = self.extract_channel(
            eeg_data, times, channels, default_channel, stim_out)

        # Extract slow wave signals
        s_order = 3
        line_free = self.filter_signal(eeg_monitor, (49, 51),
                                       s_fs, s_order, "bandstop")
        signal_delta = self.filter_signal(line_free, (0.5, 4),
                                          s_fs, s_order, "bandpass")
        signal_SO = self.filter_signal(line_free, (0.5, 2),
                                       s_fs, s_order, "bandpass")

        # Extract predictions (downstate and upstate timestamps)
        predictions = []
        exclude_pred = False
        for iPred in range(len(pred_out)):
            line = pred_out[iPred]  # Extract row
            if 'Switched channel to' in line:
                # For some subjects, we generated a pseudo pred file from
                # the stim file to satisfy the file indexing step. This
                # file can not be interpretated and shall be ignored
                exclude_pred = True
            predictions.append(line)

        if exclude_pred == True:
            predictions = []
        pred = self.extract_prediction_times(
            signal_delta, signal_SO, times, predictions, s_fs)
        pred = self.select_predictions(pred, s_fs)

        # Extract stimulations
        stimulations = []
        for iStim in range(len(stim_out)):
            line = stim_out[iStim]  # Extract row

            # The next if statement is ncomplete as it will retain in 
            # stimulations the line when it says Program quit irreversibly 
            # or timestamps of all upastates etc.
            # Should be "if '.wav' not in line"
            # But we maintain it as is since it will affect all downstream 
            # noise epoch evaluations. We will deal with this later.
            if 'Switched channel' in line:
                continue
            stimulations.append(line)

        stim = self.extract_stimulation_times(
            signal_delta, signal_SO, times, stimulations, s_fs)

        # Extract sleep/wake stagings
        wake_stagings = []
        sws_stagins = []
        for iStage in range(len(stage_out)):
            line = stage_out[iStage]  # Extract row
            if 'SWS' in line:
                sws_stagins.append(line)
            elif 'awake' in line:
                wake_stagings.append(line)

        sleep = self.extract_stagings(times, sws_stagins, wake_stagings, s_fs)

        # Build MNE data structure
        # -----------------------------------------------------------------
        channel_index = []
        channel_type = []
        channel_name = []

        for ch_name, ch_index in eeg_info['electrodes'].items():
            if "None" in ch_name:
                continue

            # For some subjects, we had poor definition of channel names.
            # We correct them here
            if ch_name == "One":
                continue  # Not used
            elif ch_name == "Two":
                continue  # Not used
            elif ch_name == "Three":
                continue  # Not used
            elif ch_name == "Four":
                ch_name = "Fp2"
            elif ch_name == "Five":
                ch_name = "Fp1"
            elif ch_name == "Six":
                ch_name = "EMG"
            elif ch_name == "Seven":
                ch_name = "VEOG"
            elif ch_name == "Eight":
                ch_name = "HEOG"

            # Add non-eeg channels
            if ch_name == "VEOG" or ch_name == "HEOG" or ch_name == "EMG":
                if ch_name == "VEOG" or ch_name == "HEOG":
                    channel_index.append(ch_index)
                    channel_type.append('eog')  # Ocular channels
                    channel_name.append(ch_name)

                if ch_name == "EMG":
                    channel_index.append(ch_index)
                    channel_type.append('emg')
                    channel_name.append(ch_name)

            else:  # Add eeg channels
                channel_index.append(ch_index)
                channel_type.append('eeg')
                channel_name.append(ch_name)

        # Add recombined stim channel
        channel_name.append('RT')
        channel_type.append('eeg')
        channel_index.append(eeg_data.shape[0])
        full_egg = np.vstack((eeg_data, np.expand_dims(eeg_monitor, 0)))

        sample_rate = float(eeg_info['sample freq'])

        info = mne.create_info(
            channel_name, ch_types=channel_type,
            sfreq=sample_rate)

        raw = mne.io.RawArray(full_egg[channel_index, :],
                              info, verbose="DEBUG")
        # The data has n_times samples and the recording is
        # n_times-1 * 1000 / sample_rate seconds long

        orig_time = datetime.strptime(eeg_info["date"], "%d-%m-%Y_%H-%M-%S")
        orig_time.replace(tzinfo=timezone.utc)
        orig_time = None
        s_duration = 0
        ms_to_s = 1000

        times_real = times  # With ms fluctuations
        samples_to_ms = int(1000/s_fs)
        times_theo = np.asarray(range(times_real.size))
        times_theo = times_theo * samples_to_ms

        # Set annotations along data

        # x_data are indices and need to be converted to times (perfect
        # times with 1000/s_fr)
        x_down = pred["x_down"] * samples_to_ms
        x_up = pred["x_up"] * samples_to_ms
        x_stim = stim["x_stim"] * samples_to_ms
        x_sws = sleep["x_sws"] * samples_to_ms
        x_wake = sleep["x_wake"] * samples_to_ms

        # Find perfect time of these elements (currently they are using
        # real timestamps)
        x_breaks = np.zeros((len(t_file_breaks)))
        for iBrk in range(len(t_file_breaks)):
            fbreak = t_file_breaks[iBrk]
            time_diffs = np.abs(times - fbreak)
            idx_time = int(np.where(time_diffs == np.min(time_diffs))[0][0])
            x_breaks[iBrk] = idx_time * samples_to_ms

        x_switches = np.zeros((len(t_switches)))
        for iSwt in range(len(t_switches)):
            fbreak = t_switches[iSwt]
            time_diffs = np.abs(times - fbreak)
            idx_time = int(np.where(time_diffs == np.min(time_diffs))[0][0])
            x_switches[iSwt] = idx_time * samples_to_ms

        # Identify down states corresponding to stimulated SOs
        x_stim_down = []
        for x_st in x_stim:
            idx_prior_down = np.where(x_down <= x_st)[0]
            if idx_prior_down.size > 0:
                x_stim_down.append(x_down[idx_prior_down[-1]])
            else:
                print("Stimulation did not have prior downstate")
        x_stim_down = np.asarray(list(set(x_stim_down)), dtype=int)

        stim["stim_down"] = x_stim_down

        down_annotations = mne.Annotations(
            onset=x_down/ms_to_s,
            duration=s_duration, description="down", orig_time=orig_time)
        up_annotations = mne.Annotations(
            onset=x_up/ms_to_s,
            duration=s_duration, description="up", orig_time=orig_time)
        stim_annotations = mne.Annotations(
            onset=x_stim/ms_to_s,
            duration=s_duration, description=stim["c_stim"], orig_time=orig_time)
        sleep_annotations = mne.Annotations(
            onset=x_sws/ms_to_s,
            duration=s_duration, description=list(sleep["verb_sws"]), orig_time=orig_time)
        wake_annotations = mne.Annotations(
            onset=x_wake/ms_to_s,
            duration=s_duration, description=list(sleep["verb_wake"]), orig_time=orig_time)
        break_annotations = mne.Annotations(
            onset=x_breaks/ms_to_s,
            duration=s_duration, description='Break', orig_time=orig_time)
        switch_annotations = mne.Annotations(
            onset=x_switches/ms_to_s,
            duration=s_duration, description='Switch', orig_time=orig_time)
        stim_down_annotations = mne.Annotations(
            onset=x_stim_down/ms_to_s,
            duration=s_duration, description='Stim_down', orig_time=orig_time)

        raw.set_annotations(
            up_annotations +
            down_annotations +
            stim_annotations +
            sleep_annotations +
            wake_annotations +
            break_annotations +
            switch_annotations +
            stim_down_annotations,
            verbose='INFO')

        Subject = {}
        Subject["mne_raw"] = raw
        Subject["times"] = times_theo
        Subject["times_real"] = times_real
        Subject["info"] = eeg_info
        Subject["pred"] = pred
        Subject["stim"] = stim
        Subject["sleep"] = sleep

        return Subject
    
    def extract_data(self, file_path, file_name, latency_shift, data_type):

        # Load data
        data_in = open(file_path + '/' + file_name, 'r')
        data_in = data_in.read()

        # Replace time stamps with shifted ones according to last file
        temp_list = data_in.split('\n')
        # Identify first data row: 4th (we reject first data rows as well
        # because of inconsistent time stamps when starting code)
        idx_first = 3
        temp_list = temp_list[idx_first:-1]  # Last one is empty line

        for iStp in range(len(temp_list)):
            line = temp_list[iStp]

            if line == '':
                pass
            idx_split = line.find(',')  # Find comma behind time stamp
            time_line = line[0:idx_split]

            try:
                time_stamp = str(float(time_line) + latency_shift)
            except:
                print('Jumped incomplete line')
                continue

            # If prediction, then time stamps of predicted upstates also need latency_shift
            if data_type == 'pred':
                idx_split2 = line.find('at ')

                if idx_split2 == -1:
                    continue  # Not found

                t_upstate = line[idx_split2+3:]
                upstate_stamp = str(float(t_upstate) + latency_shift)
                temp_list[iStp] = time_stamp + \
                    line[idx_split:idx_split2+3] + upstate_stamp
            else:
                temp_list[iStp] = time_stamp + line[idx_split:]

        # Updat latency shift
        if len(temp_list) == 0:
            return [], latency_shift  # Can happen in pred and stim files that
            # they are empty because no SO detected and stimulated
        line = temp_list[-1]
        idx_split = line.find(',')  # Find comma behind time stamp
        latency_shift = float(line[0:idx_split]) + 1  # Add 1 ms

        return temp_list, latency_shift

    def extract_hdr_info(self, header_line):

        eeg_info = json.loads(header_line)
        eeg_info['electrodes'] = json.loads(eeg_info['electrodes'])

        return eeg_info

    def extract_channel(self, eeg_signal, times, channels, default_channel, stim_info):
        # Input
        # - eeg_signal      Numpy array [channels x samples]
        # - times           Numpy array of time stamps
        # - channels        List of strings
        # - default_channel String indicating which channel should be extracted
        # - stim_info       Stim output by sleep recording indicating whether
        #                   channels have been switched
        # Output
        # - signal          Numpy vector [samples x 0]
        # - switches        List of time stamps of channel switches
        idx_default = [iC for iC in range(
            len(channels)) if default_channel in channels[iC]]
        signal_default = eeg_signal[idx_default, :][0]

        # Recompose signal if channel had been switched
        time_switched = []
        switched_to = []
        for line in stim_info:
            if 'Switched channel to' in line:
                line_list = line.split(',')
                idx_start = line.find('(')
                idx_stop = line.find(')')
                switched_to.append(line[idx_start+1:idx_stop])

                time_stamp = float(line_list[0])
                time_switched.append(int(np.where(times == time_stamp)[0]))

        signal_recomb = signal_default
        for iSwitch in range(len(time_switched)):

            if iSwitch == 0 and len(time_switched) > 1:
                time_start = time_switched[iSwitch]
                time_stop = time_switched[iSwitch+1]
                idx_chan = [iC for iC in range(
                    len(channels)) if switched_to[iSwitch] in channels[iC]]

                signal_channel = eeg_signal[idx_chan, :][0]
                signal_recomb[time_start:time_stop] = signal_channel[time_start:time_stop]

            elif iSwitch == len(time_switched)-1:
                time_start = time_switched[iSwitch]
                time_stop = eeg_signal.shape[1]
                idx_chan = [iC for iC in range(
                    len(channels)) if switched_to[iSwitch] in channels[iC]]

                signal_channel = eeg_signal[idx_chan, :][0]
                signal_recomb[time_start:time_stop] = signal_channel[time_start:time_stop]

        return signal_recomb, time_switched

    def filter_signal(self, signal, freq_range, sample_rate, filter_order, filter_type):

        b, a = scipy.signal.butter(filter_order, freq_range, btype=filter_type,
                                   fs=sample_rate)
        signal_filtered = np.transpose(scipy.signal.filtfilt(
            b, a, signal, padtype='odd', padlen=signal.size-1))

        return signal_filtered
    
    def extract_prediction_times(self, signal_delta, signal_SO, times, predictions, sfr):
        # Input
        # - signal_delta        Numpy vector [samples x 0]
        # - signal_SO           Numpy vector [samples x 0]
        # - times               Numpy vector [samples x 0]
        # - predictions         List of EazzZyLearn output lines
        # Output
        # - x and y             Numpy vector [time stamps x 0]

        # x = downstate and upstate time stamps
        # y = amplitude of delta signal at time stamps
        downstate_times = []
        upstate_times = []
        for pred in predictions:
            if len(pred) == 0:
                continue
            downstate_times.append(float(pred[0:pred.find(',')]))
            upstate_times.append(float(pred[pred.find('at ')+3:]))

        x_down = np.zeros(len(downstate_times), dtype='int')
        y_down = np.zeros(len(downstate_times), dtype='float')
        i = 0
        idx_reject = []
        for down_time in downstate_times:
            # Look to which time stamp the downstate time stamp is closest
            if down_time <= 0:
                # Happened once (S019) that last prediction had negative time
                # stamp
                idx_reject.append(i)
            time_diffs = np.abs(times - down_time)
            idx_time = int(np.where(time_diffs == np.min(time_diffs))[0][0])
            x_down[i] = idx_time
            y_down[i] = signal_delta[idx_time]

            if self.is_noisy_time_window(signal_delta, idx_time, sfr):
                idx_reject.append(i)

            i = i + 1

        # Predicted upstates
        # x = upstate timestamps (minimal distance to time stamps from EEG signal)
        # y = amplitude of SO signal at x
        x_up = np.zeros(len(upstate_times), dtype='int')
        y_up = np.zeros(len(upstate_times), dtype='float')
        i = 0
        for up_time in upstate_times:
            time_diffs = np.abs(times - up_time)
            # double [0][0] important here ins some cases
            idx_time = int(np.where(time_diffs == np.min(time_diffs))[0][0])
            x_up[i] = idx_time
            y_up[i] = signal_delta[idx_time]
            i = i + 1

        SO_stamps = {}
        SO_stamps["x_down"] = np.delete(x_down, idx_reject)
        SO_stamps["y_down"] = np.delete(y_down, idx_reject)
        SO_stamps["x_up"] = np.delete(x_up, idx_reject)
        SO_stamps["y_up"] = np.delete(y_up, idx_reject)

        print("Removed {} predictions because of non-physiological amplitudes".format(idx_reject))

        return SO_stamps
    
    def extract_stimulation_times(self, signal_delta, signal_SO, times, stimulations, sfr):
        # Input
        # - signal_delta        Numpy vector [samples x 0]
        # - signal_SO           Numpy vector [samples x 0]
        # - times               Numpy vector [samples x 0]
        # - stimulations        List of EazzZyLearn output lines
        # Output
        # - x and y             Numpy vector [time stamps x 0]
        # - cond_stim           Stimulation type

        # x = downstate and upstate time stamps
        # y = amplitude of delta signal at time stamps
        stim_times = []
        cond_stim = []
        for stim in stimulations:
            if len(stim) == 0:
                continue
            if ".wav" not in stim:
                continue
            # stim_times.append(float(stim[0:stim.find(',')]))
            stim_times.append(float(stim.split(',')[0]))
            stim_str = stim.split(',')[1]
            stim_str = stim_str.replace(' ', '')
            cond_stim.append(stim_str[:stim_str.find('.wav')])

        x_stim = np.zeros(len(stim_times), dtype='int')
        y_stim = np.zeros(len(stim_times), dtype='float')
        i = 0
        idx_reject = []
        for stim_time in stim_times:
            # Look to which time stamp the downstate time stamp is closest
            time_diffs = np.abs(times - stim_time)
            idx_time = int(np.where(time_diffs == np.min(time_diffs))[0][0])
            x_stim[i] = idx_time
            y_stim[i] = signal_delta[idx_time]

            if self.is_noisy_time_window(signal_delta, idx_time, sfr):
                idx_reject.append(i)

            i = i + 1

        Stim_stamps = {}
        Stim_stamps["x_stim"] = np.delete(x_stim, idx_reject)
        Stim_stamps["y_stim"] = np.delete(y_stim, idx_reject)
        Stim_stamps["c_stim"] = np.delete(cond_stim, idx_reject)

        print("Removed stimulations {} because of non-physiological amplitudes".format(idx_reject))

        return Stim_stamps

    def extract_stagings(self, times, sws_stagins, wake_stagings, sfr):
        # Times have to be shifted 3 seconds for wakings
        # and 30 seconds for sleep
        # and then the duration of the annotations has to be set to 3 and 30 respectively

        sws_times = []
        sws_bool = []
        for sws in sws_stagins:
            if len(sws) == 0:
                continue
            sws_times.append(float(sws.split(',')[0]))
            # Since some subjects have time stamps of 10⁻⁴s instead of ms,
            # we purposely do not shift the time stamps of the stagings,
            # even if this means, that stage timestamps are at the righter
            # edge of the time windows that were used for staging.
            sws_str = sws.split(',')[1]
            sws_str = sws_str.replace(' ', '')
            idx_endOI = sws_str.find('{')
            sws_str = sws_str[:idx_endOI]
            if 'False' in sws_str:
                sws_bool.append(0)
            elif 'True' in sws_str:
                sws_bool.append(1)
            else:
                raise Exception('Did not recognize SWS staging')

        wake_times = []
        wake_bool = []
        for awk in wake_stagings:
            if len(awk) == 0:
                continue
            wake_times.append(float(awk.split(',')[0]))
            # Since some subjects have time stamps of 10⁻⁴s instead of ms,
            # we purposely do not shift the time stamps of the stagings,
            # even if this means, that stage timestamps are at the righter
            # edge of the time windows that were used for staging.
            awk_str = awk.split(',')[1]
            awk_str = awk_str.replace(' ', '')
            idx_endOI = awk_str.find('{')
            awk_str = awk_str[:idx_endOI]
            if 'False' in awk_str:
                wake_bool.append(0)
            elif 'True' in awk_str:
                wake_bool.append(1)
            else:
                raise Exception('Did not recognize awake staging')

        x_sws = np.zeros(len(sws_times), dtype='int')
        i = 0
        idx_sws_reject = []
        for sws_time in sws_times:
            # Look to which time stamp the downstate time stamp is closest
            if sws_time <= 0:
                # Happened once (S019) that last prediction had negative time
                # stamp
                idx_sws_reject.append(i)
            time_diffs = np.abs(times - sws_time)
            idx_time = int(np.where(time_diffs == np.min(time_diffs))[0][0])
            x_sws[i] = idx_time
            i = i + 1

        x_wake = np.zeros(len(wake_times), dtype='int')
        i = 0
        idx_wake_reject = []
        for wake_time in wake_times:
            # Look to which time stamp the downstate time stamp is closest
            if wake_time <= 0:
                # Happened once (S019) that last prediction had negative time
                # stamp
                idx_wake_reject.append(i)
            time_diffs = np.abs(times - wake_time)
            idx_time = int(np.where(time_diffs == np.min(time_diffs))[0][0])
            x_wake[i] = idx_time
            i = i + 1

        y_wake = np.asarray(wake_bool)
        y_sws = np.asarray(sws_bool)

        verb_sws = np.chararray((y_sws.size),
                                itemsize=2, unicode=True)
        verb_sws[:] = 'LS'  # Light Sleep
        for iSWS in range(y_sws.size):
            if y_sws[iSWS] == 1:
                verb_sws[iSWS] = 'DS'  # Deep sleep (SWS)

        verb_wake = np.chararray((y_wake.size),
                                 itemsize=2, unicode=True)
        verb_wake[:] = 'AS'  # ASleep
        for iWK in range(y_wake.size):
            if y_wake[iWK] == 1:
                verb_wake[iWK] = 'AW'  # AWake

        # Reject non-valid time stamps
        if len(idx_sws_reject) > 0:
            idx_sws_reject = np.array(idx_sws_reject)
            x_sws = np.delete(x_sws, idx_sws_reject)
            y_sws = np.delete(y_sws, idx_sws_reject)
            verb_sws = np.delete(verb_sws, idx_sws_reject)
        if len(idx_wake_reject) > 0:
            idx_wake_reject = np.array(idx_wake_reject)
            x_wake = np.delete(x_wake, idx_wake_reject)
            y_wake = np.delete(y_wake, idx_wake_reject)
            verb_wake = np.delete(verb_wake, idx_wake_reject)

        Stage_stamps = {}
        Stage_stamps["x_wake"] = x_wake
        Stage_stamps["y_wake"] = y_wake
        Stage_stamps["verb_wake"] = verb_wake
        Stage_stamps["x_sws"] = x_sws
        Stage_stamps["y_sws"] = y_sws
        Stage_stamps["verb_sws"] = verb_sws

        return Stage_stamps

    def select_predictions(self, SO_stamps, sfr):
        # SO_stamps         Dictionnary with times stamps in ms (x) and
        #                   amplitudes in microV (y)

        tolerance_window = 100  # ms, in which duplicated downstates will be rejected

        s_tolerance_samples = tolerance_window * sfr / 1000

        x_down = SO_stamps["x_down"]
        y_down = SO_stamps["y_down"]
        x_up = SO_stamps["x_up"]
        y_up = SO_stamps["y_up"]

        idx_reject = np.array([])
        for iDown in range(len(x_down)):

            if np.in1d(iDown, idx_reject):
                # This is crucial in order to retain the ones that were
                # excluded from rejection before by "idx = np.delete(idx, 0)"
                continue

            time_window = (x_down[iDown] - s_tolerance_samples,
                           x_down[iDown] + s_tolerance_samples)

            x_down_higher = np.where(x_down > time_window[0])[0]
            x_down_lower = np.where(x_down < time_window[1])[0]

            idx = np.where(np.in1d(x_down_lower, x_down_higher))[0]

            if len(idx) == 1:
                continue
            elif len(idx) > 1:
                idx = np.delete(idx, 0)
                idx_reject = np.append(idx_reject, idx)

        idx_reject = np.unique(idx_reject)
        idx_reject = np.array(idx_reject, dtype='int32')

        # Reject elements
        y_down = np.delete(y_down, idx_reject)
        x_up = np.delete(x_up, idx_reject)
        y_up = np.delete(y_up, idx_reject)
        x_down = np.delete(x_down, idx_reject)

        SO_stamps["x_down"] = x_down
        SO_stamps["y_down"] = y_down
        SO_stamps["x_up"] = x_up
        SO_stamps["y_up"] = y_up

        return SO_stamps    
    
    def is_noisy_time_window(self, signal, timestamp, sfr):
        # Here, we look for potential noise and ignore the stim if noisy
        window_seconds      = 6 # seconds
        phy_threshold       = 300 # microvolt

        window_samples      = window_seconds * sfr

        if timestamp - window_samples/2 < 0:
            left_edge       = 0
        else:
            left_edge       = round(timestamp - window_samples/2)

        if timestamp + window_samples/2 > signal.size:
            right_edge      = signal.size
        else:
            right_edge      = round(timestamp + window_samples/2)

        if any(signal[left_edge:right_edge] < -phy_threshold) or any(signal[left_edge:right_edge] > phy_threshold):
            return True
        else:
            return False