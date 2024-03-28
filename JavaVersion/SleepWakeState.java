package com.example.bmws_app;

import android.content.Context;
import android.util.Log;

import org.apache.commons.lang3.tuple.Pair;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import brainflow.BrainFlowError;
import brainflow.DataFilter;
import brainflow.WindowOperations;

public class SleepWakeState extends Thread{

    public SleepWakeState(Context appContext){
        this.appContext = appContext;
    }

    Context appContext;

    double lastSleepStageTime;
    double lastWakeStageTime;
    int sleepStagingInterval = Parameters.SLEEP_STAGE_INTERVAL * 1000;
    int wakeStagingInterval = Parameters.WAKE_STAGE_INTERVAL * 1000;

    Boolean[] priorAwakeArray = MathTools.ones(Parameters.LOCKING_LENGTH / Parameters.WAKE_STAGE_INTERVAL);
    List<Boolean> priorAwake = new ArrayList<>(Arrays.asList(priorAwakeArray));

    int sampleRate = Parameters.SAMPLE_RATE;
    boolean isAwake = true;
    boolean isSws = false;
    Map<String, Float[]> frequencyBands = Parameters.FREQUENCY_BANDS;
    Map<String, Float> sleepThresholds = Parameters.SLEEP_THRESHOLDS;
    Map<String, Float> wakeThresholds = Parameters.WAKE_THRESHOLDS;

    Thread wakeThread = new Thread();
    Thread sleepThread = new Thread();

    void masterStageWakeAndSleep(Double[] vWake, Double[] vSleep, Float[] frequencyRange, String outputFile, double timestamp){
        boolean runWakeStaging = false;
        boolean runSleepStaging = false;

        if(timestamp >= lastWakeStageTime + wakeStagingInterval){
            lastWakeStageTime = timestamp;
            runWakeStaging = true;
        }
        if(timestamp >= lastSleepStageTime + sleepStagingInterval){
            lastSleepStageTime = timestamp;
            runSleepStaging = true;
        }

        if(runWakeStaging){
            if(!wakeThread.isAlive()){
                wakeThread = new Staging(vWake, Parameters.stagingType.IS_AWAKE, frequencyRange, outputFile, timestamp);
                wakeThread.start();
            }
            else{
                Log.i("SleepWakeState", "Skipped wake staging: wake thread still active");
            }
        }

        if(runSleepStaging){
            if(!sleepThread.isAlive()){
                sleepThread = new Staging(vWake, Parameters.stagingType.IS_SSWS, frequencyRange, outputFile, timestamp);
                sleepThread.start();
            }
            else{
                Log.i("SleepWakeState", "Skipped sleep staging: sleep thread sill active");
            }
        }
    }

    class Staging extends Thread{
        public Staging(Double[] data, Parameters.stagingType type, Float[] frequencyRange, String outputFile, double timestamp){
            this.data = data;
            this.type = type;
            this.frequencyRange = frequencyRange;
            this.outputFile = outputFile;
            this.timestamp = timestamp;
        }

        Double[] data;
        Parameters.stagingType type;
        Float[] frequencyRange;
        String outputFile;
        double timestamp;

        public void run() {
            Float[] dataAsFloat = MathTools.toFloatArray(data);
            Pair<double[], double[]> freqsPower = powerSpectrumWelch(dataAsFloat, frequencyRange, sampleRate);

            String line;
            Map<String, Float> stagingRatios;
            if(type == Parameters.stagingType.IS_AWAKE){
                line = String.format("%s, Subject awake = ", timestamp);
                stagingRatios = wakeThresholds;
            }else{
                line = String.format("%s, Subject in SWS = ", timestamp);
                stagingRatios = sleepThresholds;
            }
            String[] stagingRatiosKeys = stagingRatios.keySet().toArray(new String[0]);

            Map<String, Boolean> predictions = new HashMap<String, Boolean>();
            for (String ratioString : stagingRatiosKeys) {
                // ratioThreshold should not be null
                Float ratioThreshold = stagingRatios.get(ratioString);
                String[] bands = ratioString.split("VS");
                predictions.put(ratioString, bandRatioThresholding(freqsPower.getLeft(), freqsPower.getRight(), bands, ratioThreshold));
            }

            Boolean[] indivPredictions = predictions.values().toArray(new Boolean[0]);

            if(type == Parameters.stagingType.IS_AWAKE){
                if(MathTools.boolSum(indivPredictions) >= indivPredictions.length / 2){
                    isAwake = true;
                    line += "true";
                }else{
                    isAwake = false;
                    line += "false";
                }
                priorAwake.add(isAwake);
                priorAwake.remove(0);

                if(MathTools.boolSum(priorAwake) >= priorAwake.size() / 2 && !isAwake){
                    isAwake = true;
                    line += " (False negative)";
                }
            }else{
                if(MathTools.boolSum(indivPredictions) >= indivPredictions.length / 2){
                    isSws = true;
                    line += "true";
                }else{
                    isSws = false;
                    line += "false";
                }
            }

            Log.i("SleepWakeState", line);
            FileManager.writeFile(appContext, outputFile, line);
        }
    }

    boolean bandRatioThresholding(double[] power, double[] freqs, String[] bands, Float thresholdValue){
        boolean prediction = false;

        if(bands.length == 1){
            String band = bands[0];
            Float[] passBand = frequencyBands.get(band);

            Integer[] low = MathTools.whereLess(freqs, passBand[1]);
            Integer[] high = MathTools.whereGreater(freqs, passBand[0]);

            Boolean[] fPass = MathTools.isIn(Arrays.asList(high), Arrays.asList(low));

            // Overlap should contain all the indexes of high where fPass == true
            Integer[] overlapsSelected = MathTools.whereEqual(fPass, true);
            Integer[] overlap = MathTools.getElements(high, overlapsSelected);

            double[] bandPowersSelected = MathTools.getElements(power, overlap);
            float bandPower = MathTools.mean(bandPowersSelected);

            prediction = bandPower > thresholdValue;
        }
        else if (bands.length == 2){
            Float[] bandPower = new Float[2];
            for (int i = 0; i < 2; i++) {
                Float[] passband = frequencyBands.get(bands[i]);

                Integer[] low = MathTools.whereLess(freqs, passband[1]);
                Integer[] high = MathTools.whereGreater(freqs, passband[0]);
                Boolean[] fPass = MathTools.isIn(Arrays.asList(high), Arrays.asList(low));

                Integer[] overlapsSelected = MathTools.whereEqual(fPass, true);
                Integer[] overlap = MathTools.getElements(high, overlapsSelected);

                double[] powerSelected = MathTools.getElements(power, overlap);

                bandPower[i] = MathTools.mean(powerSelected);
            }

            float bandRatio = bandPower[0] / bandPower[1];

            prediction = bandRatio > thresholdValue;
        }

        return prediction;
    }

    Pair<double[], double[]> powerSpectrumWelch(Float[] wholeRangeSignal, Float[] frequencyRange, int sampleRate){
        int window = 4 * sampleRate;
        double[] wholeRangeSignalAsDouble = MathTools.toDoubleArray(wholeRangeSignal);
        Pair<double[], double[]> freqsPowerRaw;
        try {
            //freqsPowerRaw = DataFilter.get_psd_welch(wholeRangeSignalAsDouble, 0, 0, sampleRate, window);
            freqsPowerRaw = DataFilter.get_psd(wholeRangeSignalAsDouble, 0, wholeRangeSignalAsDouble.length, sampleRate, WindowOperations.HANNING);
        } catch (BrainFlowError e) {
            Log.i("PowerSpectrumWelch", e.msg);
            return null;
        }
        Float[] fPass = {frequencyRange[0], frequencyRange[1]};
        Integer[] fHighPass = MathTools.whereGreaterEqual(freqsPowerRaw.getLeft(), fPass[0]);
        Integer[] fLowPass = MathTools.whereLessEqual(freqsPowerRaw.getLeft(), fPass[1]);
        Boolean[] fPassband = MathTools.isIn(Arrays.asList(fHighPass), Arrays.asList(fLowPass));

        Integer[] fPassbandSelected = MathTools.whereEqual(fPassband, true);
        Integer[] findX = MathTools.getElements(fHighPass, fPassbandSelected);

        double[] powerSelected = MathTools.getElements(freqsPowerRaw.getLeft(), findX);
        double[] freqsSelected = MathTools.getElements(freqsPowerRaw.getRight(), findX);

        Pair<double[], double[]> freqsPower = Pair.of(freqsSelected, powerSelected);
        return freqsPower;
    }
}
