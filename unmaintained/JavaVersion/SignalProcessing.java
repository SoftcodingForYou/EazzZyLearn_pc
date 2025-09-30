package com.example.bmws_app;

import android.util.Log;
import com.github.psambit9791.jdsp.filter.Butterworth;

import org.json.JSONException;
import org.json.JSONObject;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

public class SignalProcessing {

    // Define channel to use throughout pipeline
    // ---------------------------------------------------------------------------------------------
    int currentChannel              = Parameters.IDX_ELEC;
    String[] channels               = Parameters.ELEC.keySet().toArray(new String[0]);

    // Filter parameters
    // ---------------------------------------------------------------------------------------------
    int filterOrder                 = Parameters.FILTER_ORDER;
    Butterworth butterworth         = new Butterworth(Parameters.SAMPLE_RATE);
    Float[] lineNoiseFB             = Parameters.FREQUENCY_BANDS.get("LineNoise");
    Float[] wholeFB                 = Parameters.FREQUENCY_BANDS.get("Whole");
    Float[] deltaFB                 = Parameters.FREQUENCY_BANDS.get("Delta");
    Float[] slowDeltaFB             = Parameters.FREQUENCY_BANDS.get("SlowDelta");

    // Define parameters of signal vectors
    // ---------------------------------------------------------------------------------------------
    int mainBufferLength            = Parameters.MAIN_BUFFER_LENGTH;
    int deltaBufferLength           = Parameters.DELTA_BUFFER_LENGTH;
    int thresholdBufferLength       = Parameters.THRESHOLD_BUFFER_LENGTH;
    int sleepBufferLength           = Parameters.SLEEP_BUFFER_LENGTH;
    int wakeBufferLength            = Parameters.RESPONSE_BUFFER_LENGTH;
    int padLen                      = mainBufferLength / 10 - 1;

    Double[] filterSignal(Double[] signal, Float lowCutoff, Float highCutoff, Boolean passband) {

        // Replacement method for lfilter and filtfilt will be butterworth
        Double[] paddedSignal       = padSignal(signal);
        //Log.i("FilterSignalPaddedSignal", Arrays.toString(paddedSignal));
        double[] paddedSignalPrimitive  = MathTools.toDoubleArray(paddedSignal);
        double[] filteredSignal;
        if (passband == true) {
            filteredSignal = butterworth.bandPassFilter(paddedSignalPrimitive, filterOrder, lowCutoff, highCutoff);
        }else{
            filteredSignal = butterworth.bandStopFilter(paddedSignalPrimitive, filterOrder, lowCutoff, highCutoff);
        }
        Double[] filteredSignalBoxed= MathTools.toDoubleArray(filteredSignal);
        Double[] output             = unpadSignal(filteredSignalBoxed);
        return output;
    }

    // Pads the signal to apply the band pass filter later
    Double[] padSignal(Double[] signal){
        //Log.i("PadSignalBeforePadding", Arrays.toString(signal));
        int totalLength             = signal.length + padLen;
        //Log.i("PadSignalBeforePadding", String.format("signal length: %d, padding length: %d, total length: %d", signal.length, padLen, totalLength));
        Double[] output             = new Double[totalLength];
        Double[] selectedSignalSegment = Arrays.copyOfRange(signal, 0, padLen);
        //Log.i("PadSignalSignalSegment", Arrays.toString(selectedSignalSegment));
        Collections.reverse(Arrays.asList(selectedSignalSegment));
        //Log.i("PadSignalReversedSignalSegment", Arrays.toString(selectedSignalSegment));

        // TODO: Double-check the correctness of the vector generation by "arraycopy"
        if (padLen >= 0) System.arraycopy(selectedSignalSegment, 0, output, 0, padLen);
        System.arraycopy(signal, 0, output, padLen, signal.length);
        return output;
    }

    Double[] unpadSignal(Double[] paddedSignal){
        Double[] output             = Arrays.copyOfRange(paddedSignal, padLen, paddedSignal.length);
        return output;
    }

    Double[][] masterExtractSignal(ArrayDeque<Integer>[] buffer){

        // Get rid of non-used channels inside signal buffer
        // -----------------------------------------------------------------------------------------
        // ALL downstream steps performed on the one used channel.
        //Log.i("MasterExtractSignal", String.format("Current channel: %s", currentChannel));
        ArrayDeque<Integer> vRaw    = buffer[currentChannel];
        //Log.i("MasterExtractSignal", vRaw.toString());

        // Extract signals in frequency bands of interest
        // -----------------------------------------------------------------------------------------
        Double[] vRawDouble         = MathTools.toDoubleArray(vRaw);
        Double[] vCleanFiltFilt     = filterSignal(vRawDouble, lineNoiseFB[0], lineNoiseFB[1], false);
        Double[] vFilteredWhole     = filterSignal(vCleanFiltFilt, wholeFB[0], wholeFB[1], true);
        Double[] vFilteredDelta     = filterSignal(vCleanFiltFilt, deltaFB[0], deltaFB[1], true);
        Double[] vFilteredSlowDelta = filterSignal(vCleanFiltFilt, slowDeltaFB[0], slowDeltaFB[1], true);

        // Cur the signal vectors into different lengths FROM THE RIGHT EDGE (most recent samples)
        // -----------------------------------------------------------------------------------------
        Double[] vWake              = MathTools.extractArray(vFilteredWhole, wakeBufferLength);
        Double[] vSleep             = MathTools.extractArray(vFilteredWhole, sleepBufferLength);
        Double[] vDelta             = MathTools.extractArray(vFilteredDelta, deltaBufferLength);
        Double[] vSlowDelta         = MathTools.extractArray(vFilteredSlowDelta, deltaBufferLength);

        // Gang together signal vectors in map
        // -----------------------------------------------------------------------------------------
        Double[][] output           = new Double[][]{vWake, vSleep, vFilteredDelta, vDelta, vSlowDelta};
        return output;
    }

    // Currently unused method, but necessary in future for automated channel switches if signal
    // quality of current one becomes bad.
    void switchChannel(int newChannel){
        if (currentChannel != newChannel){
            Log.i("ChannelSwitch", String.format("From channel %s, to %s", currentChannel, newChannel));
            currentChannel = newChannel;
        }
    }
}
