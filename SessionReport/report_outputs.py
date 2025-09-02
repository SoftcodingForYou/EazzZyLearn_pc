# Prepare userland
import os
import numpy as np
from matplotlib.colors import Normalize
import scipy.signal
import scipy.io
from lspopt import spectrogram_lspopt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Fixes plotting issues on GNU/Linux
import numpy.matlib as mtl
import mne
from var_storage import VariableOnDisk
from datetime import datetime

class GenerateOutputs():

    def __init__(self, subject, input_dir, stim_range):
        self.subject            = subject
        self.sfr                = int(subject["info"]["sample freq"])
        self.subject_name       = subject["info"]["subject name"].replace("_", " ")
        self.recording_date     = datetime.strptime(subject["info"]["date"], "%d-%m-%Y_%H-%M-%S")
        self.input_dir          = input_dir
        self.stim_range         = [round(s) for s in stim_range]
        self.save_path          = os.path.join(self.input_dir, "Report")
        self.output_template    = os.path.join(os.path.dirname(__file__), "Report_template.svg")

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def basic_sleep_metrics(self):

        y_sws                   = self.subject["sleep"]["y_sws"]
        x_sws                   = self.subject["sleep"]["x_sws"]
        y_wake                  = self.subject["sleep"]["y_wake"]
        x_wake                  = self.subject["sleep"]["x_wake"]

        # x values are samples --> sample * 1000 / sfr = ms
        x_wake                  = x_wake * 1000 / self.sfr
        x_sws                   = x_sws * 1000 / self.sfr

        ezl_wake                = []
        idx_processed           = np.array([])
        idx_first               = np.where(x_wake >= 15000)[0][0]
        for iWk in range(idx_first,x_wake.size):
            if iWk in idx_processed:
                continue
            win_start           = x_wake >= x_wake[iWk] - 15000
            win_stop            = x_wake <= x_wake[iWk] + 30000
            # 30s since 15s current score window and the 15s of the left edge
            # of the next one
            idx_window          = np.logical_and(win_start, win_stop)
            ezl_wake.append(int(round(np.median(y_wake[idx_window]))))
            idx_processed       = np.hstack((idx_processed, np.where(idx_window)[0]))
        y_wake = np.array(ezl_wake)

        # # Most commonly sleep/wake staging is performed every 30s on 30s
        # # chunks. Here however, we stage:
        # # - Wake on 3s chunks every second
        # # - SWS on 30s chunks every 5 seconds
        # # and we should therefore expect fluctuations in the scoring. We 
        # # smooth the outcome here by a median filter:
        # # - Wake        kernel size = 30
        # # - SWS         kernel size = 6
        # sws_kernel      = 7
        wake_kernel     = 31

        # y_sws           = np.asarray(
        #     scipy.signal.medfilt(y_sws, sws_kernel), dtype='int')
        y_wake          = np.asarray(
            scipy.signal.medfilt(y_wake, wake_kernel), dtype='int')
        # y_wake                  = np.array(ezl_wake)
        y_sws                   = y_sws[0:-1:6]

        # plt.figure()
        # plt.plot(y_wake)
        # plt.plot(y_sws)
        # plt.show()

        if y_wake.size < y_sws.size:
            y_sws = y_sws[:y_wake.size]
        elif y_wake.size > y_sws.size:
            y_wake = y_wake[:y_sws.size]

        y_deep_sleep = np.zeros((y_sws.size)) 
        y_deep_sleep[np.logical_and(y_sws == 1, y_wake == 0)] = 1

        y_hypnogram                     = np.ones((y_deep_sleep.size)) # ones are "Other"
        y_hypnogram[y_wake == 1]        = 2 # twos are "Deep sleep"
        y_hypnogram[y_deep_sleep == 1]  = 0 # zeros are "Awake"


        # 1. Hypnogram
        self.generate_hypnogram(y_hypnogram)

        # 2. Recording time
        self.calculate_recording_time()

        # 3. Time asleep vs total recording time
        self.calculate_percentage_asleep()

        # 4. Percentage in deep sleep
        self.calculate_percentage_deep_sleep()

    def generate_hypnogram(self, y_hypnogram):

        online_signal           =  self.get_raw_data()

        hypno=None
        win_sec=30
        fmin=0.5
        fmax=25
        trimperc=2.5
        cmap="RdBu_r"
        vmin=None
        vmax=None
        # Calculate multi-taper spectrogram
        data                    = online_signal.get_data("RT")[0]
        nperseg                 = int(win_sec * self.sfr)
        assert data.size > 2 * nperseg, "Data length must be at least 2 * win_sec."
        f, t, Sxx = spectrogram_lspopt(data, self.sfr, nperseg=nperseg, noverlap=0)
        Sxx = 10 * np.log10(Sxx)  # Convert uV^2 / Hz --> dB / Hz

        # Select only relevant frequencies (up to 30 Hz)
        good_freqs = np.logical_and(f >= fmin, f <= fmax)
        Sxx = Sxx[good_freqs, :]
        f = f[good_freqs]
        t /= 3600  # Convert t to hours

        # Normalization
        if vmin is None:
            vmin, vmax = np.percentile(Sxx, [0 + trimperc, 100 - trimperc])
        norm = Normalize(vmin=vmin, vmax=vmax)

        # Extract stimulations
        x_stim          = self.subject["stim"]["x_stim"]

        samples_to_ms   = 1000/self.sfr
        ms_to_min       = 1/60000
        min_to_hrs      = 1/60   

        mne_raw                 = self.get_raw_data()
        # Filter real-time channel
        mne_raw.pick(['RT'])
        raw_signal              = mne_raw.load_data()
        raw_signal              = raw_signal._data   

        filt_order              = 3
        freq_range_whole        = [0.1, 45]
        freq_range_delta        = [0.5, 4]
        freq_range_slowdelta    = [0.5, 2]
        freq_range_noise        = [49, 51]
        line_free               = self.filter_signal(raw_signal,
                                    freq_range_noise, self.sfr, filt_order, 'stop')
        whole_signal            = self.filter_signal(np.transpose(line_free),
                                    freq_range_whole, self.sfr, filt_order, 'bandpass')
        delta_signal            = self.filter_signal(np.transpose(whole_signal),
                                    freq_range_delta, self.sfr, filt_order, 'bandpass')

        # Plot the results
        # ---------------------------------------------------------------------
        # Adapt lengths of vectors
        idx_end = min([Sxx.shape[1], y_hypnogram.size])

        last_signal_time    = y_hypnogram.size*30*self.sfr
        subj_times          = range(0, last_signal_time)
        subj_times          = np.array([x * 1000 / self.sfr for x in list(subj_times)])
        delta_signal        = delta_signal[0:last_signal_time]
        x_stim              = np.delete(x_stim, x_stim > last_signal_time)

        fig, ax = plt.subplots(2, 1, figsize=(9, 4))

        idx_ax = 0
        # Plot1 spectrogram of RT channel using yasa.plot_sprectrogram()
        im = ax[idx_ax].pcolormesh(t[0:idx_end], f, Sxx[:,0:idx_end], 
                            norm=norm, cmap=cmap, antialiased=True, shading="auto")
        ax[idx_ax].set_xlim(0, t[0:idx_end].max())
        ax[idx_ax].set_ylabel("Frequency [Hz]")
        ax[idx_ax].set_xlabel('')
        ax[idx_ax].set_xticklabels('')
        ax[idx_ax].legend(['Time-dependent spectrogram'], loc='upper right')
        cbar = fig.colorbar(im, ax=ax[0], shrink=0.95, fraction=0.1, aspect=25, location='top')
        cbar.ax.set_xlabel("Log Power (dB / Hz)", rotation=0, labelpad=2)

        idx_ax = 1
        # Plot4 EZL sleep scoring (deep sleep two step only)
        ax[idx_ax].plot(t[0:idx_end], y_hypnogram[0:idx_end], linewidth=2, color='k')
        # for iSc in range(idx_end):
        #     ax[2].scatter(t[iSc], y_hypnogram[iSc], s=15, color=[0, 0, 0])
        ax[idx_ax].set_xlim(0, t[0:idx_end].max())
        ax[idx_ax].set_xlabel("Time [hrs]")
        ax[idx_ax].set_ylim((-0.25, 2.25))
        # ax[idx_ax].set_ylabel('Hypnogram')
        ax[idx_ax].set_yticks([0, 1, 2])
        ax[idx_ax].set_yticklabels(['Deep sleep', 'Other', 'Awake'])
        ax[idx_ax].legend(['Hypnogram'], loc='upper right')

        plt.savefig(os.path.join(self.save_path, 'hypnogram.png'), bbox_inches='tight') # plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(9, 2))

        idx_ax = 0
        ax.plot(subj_times * ms_to_min * min_to_hrs, delta_signal[0:last_signal_time], linewidth=0.5, color=[0.2, 0.2, 0.7])
        ax.scatter(subj_times[x_stim] * ms_to_min * min_to_hrs, mtl.repmat(0, x_stim.size, 1), 15, 'k', zorder=2000)
        ax.set_ylim((-1.5*np.std(delta_signal[0:last_signal_time]), 1.5*np.std(delta_signal[0:last_signal_time])))
        ax.set_xlim((subj_times[0]* ms_to_min * min_to_hrs, subj_times[-1]* ms_to_min * min_to_hrs))
        ax.set_xlabel('')
        ax.set_ylabel('Amplitude (uV)')
        ax.set_xlabel("Time [hrs]")
        ax.legend(['Delta component', 'Sound presentation'], loc='upper right')

        plt.savefig(os.path.join(self.save_path, 'delta_component.png'), bbox_inches='tight') # plt.show()


    def calculate_recording_time(self):
        ms_to_hrs = 1 / 3600000
        self.recording_time_hrs = (self.subject["times_real"][-1] - self.subject["times_real"][0]) * ms_to_hrs

    def calculate_percentage_asleep(self):
        percentage_awake = sum(self.subject["sleep"]["y_wake"]) / self.subject["sleep"]["y_wake"].size * 100
        self.percentage_asleep = round(100 - percentage_awake, 1)

    def calculate_percentage_deep_sleep(self):
        proportion_awake = sum(self.subject["sleep"]["y_wake"]) / self.subject["sleep"]["y_wake"].size
        length_sleep = self.subject["sleep"]["y_sws"].size * (1 - proportion_awake)
        self.percentage_deep_sleep = round(sum(self.subject["sleep"]["y_sws"]) / length_sleep * 100, 1)

    def filter_signal(self, signal, freq_range, sample_rate, filter_order, filter_type):

        b, a = scipy.signal.butter(filter_order, freq_range, btype=filter_type,
                                   fs=sample_rate)
        signal_filtered = np.transpose(scipy.signal.filtfilt(
            b, a, signal, padtype='odd', padlen=signal.size-1))

        return signal_filtered

    def get_raw_data(self):
        # Extra function to avoid pointers
        data_storage = VariableOnDisk()
        data_storage.set("temp", self.subject["mne_raw"])
        data = data_storage.get("temp")
        print(id(self.subject["mne_raw"]))
        print(id(data))
        return data
    
    def stimulation_analysis(self):

        pred                    = self.subject["pred"]
        stim                    = self.subject["stim"]
        times                   = self.subject["times"]
        info                    = self.subject["info"]
        mne_raw                 = self.get_raw_data()
        # Filter real-time channel
        mne_raw.pick(['RT'])
        raw_signal              = mne_raw.load_data()._data[
            [c for c in range(len(mne_raw.ch_names)) if mne_raw.ch_names[c] == "RT"], :]

        sfr                     = int(info["sample freq"])
        filt_order              = 3
        freq_range_whole        = [0.1, 45]
        freq_range_delta        = [0.5, 4]
        freq_range_slowdelta    = [0.5, 2]
        freq_range_noise        = [49, 51]
        line_free               = self.filter_signal(raw_signal,
                                    freq_range_noise, self.sfr, filt_order, 'stop')
        whole_signal            = self.filter_signal(np.transpose(line_free),
                                    freq_range_whole, self.sfr, filt_order, 'bandpass')
        delta_signal            = self.filter_signal(np.transpose(whole_signal),
                                    freq_range_delta, self.sfr, filt_order, 'bandpass')
        SO_signal               = self.filter_signal(np.transpose(whole_signal),
                                    freq_range_slowdelta, self.sfr, filt_order, 'bandpass')

        # 1. Average signal around stimulation
        print("*** Consider changing grand average generation on stim[\"x_stim_down\"] instead of pred[\"x_down\"]")
        self.grand_average(delta_signal, pred["x_down"], times, self.sfr)
        
        # 2. Number of stimulations
        self.stimulation_scaled()
        
        # 3. Used cue sound
        self.extract_cue_information()

        # 4. Time-frequency around stimulation
        self.stim_time_freq()

        # 5. Number of slow oscillations
        self.extract_number_so()

    def extract_number_so(self):
        self.number_so = self.subject["pred"]["x_down"].size
        
    def grand_average(self, delta_signal, x_down, times, samplingrate):

        before              = 1  # seconds
        after               = 1  # seconds
        x_times             = range(-before*samplingrate, after*samplingrate)
        x_times             = [x * 1000 / samplingrate for x in list(x_times)]

        SlowOsc             = np.zeros((len(x_down), (before + after) * samplingrate))
        DownAmp             = np.zeros(len(x_down))
        Reject              = np.zeros(len(x_down))
        for iDown in range(len(x_down)):

            # # x_down are actual timestamps, but we need indices. Easily
            # # convertable since timestamps are considered to be perfect
            # # ([0, 4, 8, 12, ...])
            # idx_down        = np.where(times == x_down[iDown])[0][0]

            start           = int(x_down[iDown] - before * samplingrate)
            stop            = int(x_down[iDown] + after * samplingrate)

            if stop > delta_signal.size:
                Reject[iDown] = 1
                continue

            SlowOsc[iDown, :] = np.squeeze(delta_signal[start:stop])
            DownAmp[iDown]    = delta_signal[x_down[iDown]]

        # Delete zeros
        SlowOsc             = SlowOsc[np.where(Reject == 0)[0], :]

        # Reject outliers
        avgAmp              = np.zeros((SlowOsc.shape[0]))
        for iSO in range(SlowOsc.shape[0]):
            avgAmp[iSO]     = np.mean(abs(SlowOsc[iSO, :]))

        idx_retain          = avgAmp < np.median(avgAmp) + 3*np.std(avgAmp)
        SlowOsc             = SlowOsc[idx_retain, :]

        AvgOsc              = np.mean(SlowOsc, axis=0)

        plt.figure(figsize=(3, 2))
        for iPlt in range(SlowOsc.shape[0]):
            plt.plot(x_times, SlowOsc[iPlt], color='k')
        plt.plot(x_times, AvgOsc, color='r')
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude (uV)")
        plt.savefig(os.path.join(self.save_path, 'grand_average.png'), bbox_inches='tight') # plt.show()

    def stimulation_scaled(self):
        
        avg_stims = np.mean(self.stim_range)
        s_stims = round(len(self.subject["stim"]["x_stim"]) / self.recording_time_hrs) # per hour of recording
        x = np.arange(np.min(self.stim_range), np.max(self.stim_range), 1)
        y = np.ones(x.size)

        if s_stims > np.max(self.stim_range):
            x_stims = np.max(self.stim_range)
            i_color = -1
        elif s_stims < np.min(self.stim_range):
            x_stims = np.min(self.stim_range)
            i_color = 0
        else:
            x_stims = s_stims
            i_color = s_stims-np.min(self.stim_range)

        colors = plt.cm.jet(np.linspace(0,1,x.size))

        fig, ax = plt.subplots(1, 1, figsize=(4, 1))
        for i in range(x.size):
            ax.scatter(x[i], y[i], c=colors[i, :], marker="s", s=25)
        ax.scatter(x_stims, 1, c=colors[i_color, :], edgecolor='k', s=100)
        ax.text(x_stims, 1+0.02, str(s_stims))
        ax.scatter(avg_stims, 1, c=colors[round(avg_stims)-np.min(self.stim_range), :], edgecolor='k', s=100)
        ax.text(avg_stims-40, 1-0.04, "".join(("Average = ", str(round(avg_stims)))))

        plt.axis('off')
        plt.savefig(os.path.join(self.save_path, 'stims_scaled.png'), bbox_inches='tight') # plt.show()

    def extract_cue_information(self):
        self.used_cue = self.subject["info"]["chosencue"]

    def stim_time_freq(self):

        v_baseline              = [-1, 2] # seconds
        v_window                = [-2, 3] # seconds
        f_step                  = 0.1
        freq_range              = (9, 26)
        channels                = ['RT'] # ['Fp1', 'Fp2'] 
        raw                     = self.get_raw_data()
        raw_plot                = self.get_raw_data()

        # Filter data ---------------------------------------------------------
        raw_plot.notch_filter(50, picks='all', verbose='WARNING')
        raw_plot.filter(l_freq=0.5, h_freq=2, picks='eeg', verbose='WARNING')

        # Extract events ------------------------------------------------------
        events, event_id    = mne.events_from_annotations(raw, verbose=False)
        # events            = list where [onset time, duration, event_id.value()]

        # Combine cue stimulations and control stimulations in respective ids 
        # and replace mne-generated ones
        event_list          = list(event_id.keys())

        id_stim = []
        for event in event_list:
            if event == self.subject["info"]["chosencue"]:
            # if event == "Stim_down":
                id_stim = event_id[event]
        
        # Slice data into Place and Cue epochs
        # ---------------------------------------------------------------------
        freqs                   = np.arange(freq_range[0], freq_range[1], f_step)

        if len(self.subject["stim"]["x_stim"]) == 0:
            t                   = np.arange(v_window[0], v_window[-1], 0.10)
            subjTFR_cue         = np.zeros((freqs.size-1, t.size))
            f                   = freqs
        else:
            epochs_cue          = mne.Epochs(raw, events, 
                tmin=v_window[0], tmax=v_window[1], event_id=id_stim,
                preload=True, event_repeated='error',
                baseline = None, picks = 'all',
                verbose='WARNING')

            epochs_plot          = mne.Epochs(raw_plot, events, 
                tmin=v_window[0], tmax=v_window[1], event_id=id_stim,
                preload=True, event_repeated='error',
                baseline = None, picks = 'all',
                verbose='WARNING')
            subj_avg                = epochs_plot.average(picks=[c for c in epochs_plot.ch_names if c in channels])
            subj_wf_cue             = np.mean(subj_avg.data, 0)

            # Scale amplitude
            subj_wf_cue             = (subj_wf_cue - min(freq_range)) / (max(freq_range) - min(freq_range))
            subj_wf_cue             = subj_wf_cue + max(freq_range) - min(freq_range)
        
            # Time-frequency extraction
            # ---------------------------------------------------------------------
            
            cycles                  = freqs / 2
            print('Computing TF for Cue epochs ...')
            chanTF_epochs_cue       = self.extract_tf_matrix(epochs_cue, freqs, cycles)
            print("Done!")

            # Normalization over epochs
            print('Normalizing TF matrices ...')
            normTF_epochs_cue       = self.normalize_tf_matrices(chanTF_epochs_cue, v_baseline)
            print("Done!")

            subjTFR_cue, f, t       = self.grand_average_tf(normTF_epochs_cue, channels)

        # Plot result: Average time-frequency response to stimulation
        # -------------------------------------------------------------------------
        clim                = (-3, 3)
        v_plot              = (-1, 2) # seconds
        v_freqs             = (9, 26)

        t_0                 = np.where(t >= v_plot[0])[0][0]
        t_end               = np.where(t <= v_plot[-1])[0][-1]
        t                   = t[t_0:t_end]
        f_low               = np.where(f >= v_freqs[0])[0][0]
        f_high              = np.where(f <= v_freqs[1])[0][-1]
        f                   = f[f_low:f_high]

        fig             = plt.figure(figsize=(5, 2))
        ax              = plt.subplot(111)
        ax.set_ylabel('Frequency (Hz)')
        ax.set_xlabel('Time (s)')
        plt_ctl = ax.pcolormesh(t, f, subjTFR_cue[f_low:f_high, t_0:t_end], cmap='jet')
        fig.colorbar(plt_ctl, ax=ax, label='Magnitude (Z)')
        ax.plot([0, 0], [f[0], f[-1]], 'k')
        plt.savefig(os.path.join(self.save_path, 'time_frequency.png'), bbox_inches='tight') # plt.show()

    def extract_tf_matrix(self, epochs, freqs, cycles):
            
        chanTF_epochs       = mne.time_frequency.tfr_morlet(epochs, freqs,
                                                        n_cycles = cycles,
                                                        picks='eeg', 
                                                        average=False,
                                                        return_itc=False)
        # chanTF_epochs is 4D matrix [trials x channels x freqs x times]
        # Important: Both from "epochs" as well as from "chanTF_epochs", the 
        # bad epochs marked by idx_bad are completely removed from the object

        return chanTF_epochs

    def normalize_tf_matrices(self, chanTF_epochs, v_baseline):

        # chanTF_epochs is 4D matrix [trials x channels x freqs x times]

        t                       = chanTF_epochs.times
        idx_start               = np.where(t >= v_baseline[0])[0][0]
        idx_stop                = np.where(t <= v_baseline[-1])[0][-1]
        f                       = chanTF_epochs.freqs

        # Apply baselining (on individual epochs)
        # ---------------------------------------------------------------------
        # ‘mean’ | ‘ratio’ | ‘logratio’ | ‘percent’ | ‘zscore’ | ‘zlogratio’
        chanTF_norm             = chanTF_epochs.apply_baseline(v_baseline, mode = 'zscore')

        # Apply baselining: Z-score (on common baseline)
        # ---------------------------------------------------------------------
        # for iChan in range(chanTF_epochs.data.shape[1]):
        #     for iFreq in range(f.size):
        #         com_baseline    = chanTF_epochs.data[:, iChan, iFreq, idx_start:idx_stop].ravel()
        #         avg_baseline    = np.mean(com_baseline)
        #         std_baseline    = np.std(com_baseline)

        #         for iTrl in range(chanTF_epochs.data.shape[0]):                    
        #             chanTF_epochs.data[iTrl, iChan, iFreq, :] = (
        #                 chanTF_epochs.data[iTrl, iChan, iFreq, :] -
        #                 avg_baseline) / std_baseline
        # chanTF_norm = chanTF_epochs

        return chanTF_norm 
        
    def grand_average_tf(self, chanTF_epochs, channels):

        chanTF  = chanTF_epochs.pick(picks=channels)
        chanTF  = chanTF.average(method='median', dim='epochs', copy=False)
        
        f       = chanTF.freqs
        t       = chanTF.times

        # Average over channels
        avgTFR  = np.mean(chanTF.data, 0)

        return avgTFR, f, t
    
    def output_svg(self):
        with open(self.output_template, encoding='utf8') as f:
            lines = f.readlines()

        iL = 0
        set_parameters = {
            "set_username":             False,
            "set_date":                 False,
            "set_hypnogram":            False,
            "set_delta_component":      False,
            "set_grand_average":        False,
            "set_time_frequency":       False,
            "set_stimulation_scale":    False,
            "set_recording_time":       False,
            "set_sleep_proportion":     False,
            "set_deep_sleep_prop":      False,
            "set_total_sos":            False,
            "set_number_and_name_cue":  False,
        }

        # Important: Image lines won't be replaced because they are pointing already to images 
        # placed in the same folder as the report file
        for line in lines:

            # Set user name
            if "User: {}" in line:
                line = line.replace("User: {}", "User: {}".format(self.subject_name))
                lines[iL] = line
                set_parameters["set_username"] = True
            
            # Set date
            if "Date: {}" in line:
                line = line.replace("Date: {}", "Date: {}".format(datetime.strftime(self.recording_date, "%B %d, %Y (%H:%M)")))
                lines[iL] = line
                set_parameters["set_date"] = True

            # Set hypnogram
            if 'hypnogram.png' in line and 'xlink:href' in line:
                line = line.replace("hypnogram.png", "{}".format(os.path.join(self.save_path, "hypnogram.png")))
                lines[iL] = line
                set_parameters["set_hypnogram"] = True

            # Set Slow Osc. grand average
            if 'grand_average.png' in line and 'xlink:href' in line:
                line = line.replace("grand_average.png", "{}".format(os.path.join(self.save_path, "grand_average.png")))
                lines[iL] = line
                set_parameters["set_grand_average"] = True

            # Set time-frequency
            if 'time_frequency.png' in line and 'xlink:href' in line:
                # line = line.replace("time_frequency.png", "{}".format(os.path.join(self.save_path, "time_frequency.png")))
                lines[iL] = line
                set_parameters["set_time_frequency"] = True

            # Set stimulation scale
            if 'stims_scaled.png' in line and 'xlink:href' in line:
                line = line.replace("stims_scaled.png", "{}".format(os.path.join(self.save_path, "stims_scaled.png")))
                lines[iL] = line
                set_parameters["set_stimulation_scale"] = True

            # Set delta component
            if 'delta_component.png' in line and 'xlink:href' in line:
                line = line.replace("delta_component.png", "{}".format(os.path.join(self.save_path, "delta_component.png")))
                lines[iL] = line
                set_parameters["set_delta_component"] = True

            # Set recording time
            if "recorded for {} hours" in line:
                line = line.replace("recorded for {} hours", "recorded for {} hours".format(round(self.recording_time_hrs, 2)))
                lines[iL] = line
                set_parameters["set_recording_time"] = True

            # Set sleep proportion
            if "asleep during {}" in line:
                line = line.replace("asleep during {}", "asleep during {}".format(self.percentage_asleep))
                lines[iL] = line
                set_parameters["set_sleep_proportion"] = True

            # Set sleep proportion
            if "asleep was {}" in line:
                line = line.replace("asleep was {}", "asleep was {}".format(self.percentage_deep_sleep))
                lines[iL] = line
                set_parameters["set_deep_sleep_prop"] = True

            # Set total amount of SOs
            if "total of {} Delta waves" in line:
                line = line.replace("total of {} Delta waves", "total of {} Delta waves".format(self.number_so))
                lines[iL] = line
                set_parameters["set_total_sos"] = True

            # Set number of stimulated SOs and cue
            if "Stimulated were {} of them with the sound “{}”" in line:
                line = line.replace("Stimulated were {} of them with the sound “{}”",
                             "Stimulated were {} of them with the sound “{}”".format(len(self.subject["stim"]["x_stim"]), self.used_cue))
                lines[iL] = line
                set_parameters["set_number_and_name_cue"] = True

            iL += 1

        if len([el for el in set_parameters.values() if el == False]) > 0:
            raise Exception("One or multiple parameters were not set in output file")

        with open(os.path.join(
            self.save_path,
            "Session Report " + self.subject_name + " {}".format(datetime.strftime(self.recording_date, "%d-%B-%Y")) + ".svg"), 'w', encoding='utf8') as f:
            f.write("".join(lines))