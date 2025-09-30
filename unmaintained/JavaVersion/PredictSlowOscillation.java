package com.example.bmws_app;
import android.content.Context;
import android.util.Log;

import java.util.Arrays;

public class PredictSlowOscillation {

    public PredictSlowOscillation(Context appContext){
        this.appContext = appContext;
    }

    Context appContext;

    Float stimAtStamp = null;
    int defaultThreshold = Parameters.DEFAULT_THRESHOLD;
    int artifactThreshold = Parameters.NON_PHYSIOLOGICAL_THRESHOLD;
    float throwMultiplier = Parameters.THROW_MULTIPLICATION;

    double setThreshold(Double[] thresholdArray) {
        double minAmplitude = MathTools.min(thresholdArray);

        double adaptiveThreshold;
        if (minAmplitude < defaultThreshold) {
            adaptiveThreshold = minAmplitude;
        }else{
            adaptiveThreshold = defaultThreshold;
        }
        return adaptiveThreshold;
    }

    Double[][] extractSlowOscillationOnset(Double[] deltaArray, Double[] slowDeltaArray) {
        Integer[] signVector = MathTools.sign(deltaArray);
        Integer[] diffVector = MathTools.diff(signVector);

        Integer[] idxP2nAll = MathTools.whereEqual(diffVector, -2);

        if (idxP2nAll.length == 0) {
            return new Double[][]{{null},{null},{null}};
        }
        double idxP2n = idxP2nAll[idxP2nAll.length - 1];

        Integer[] idxN2pAll = MathTools.whereEqual(diffVector, 2);

        if (idxN2pAll.length != 0 && idxN2pAll[idxN2pAll.length - 1] > idxP2n){
            return new Double[][]{{null},{null},{null}};
        }

        Double[][] output = {{0.0},{0.0},{0.0}};
        output[0] = Arrays.copyOfRange(deltaArray, (int) idxP2n, deltaArray.length);
        output[1] = Arrays.copyOfRange(slowDeltaArray, (int) idxP2n, slowDeltaArray.length);
        output[2] = new Double[]{idxP2n};

        return output;
    }

    // This method is the same as downstate_validation but with a correct coding name, as it returns a bool
    boolean downstateValid(Double[] soOnsetArray, double threshold) {
        int idxDownstate = MathTools.argmin(soOnsetArray);
        Double[] postDownSignal = Arrays.copyOfRange(soOnsetArray, idxDownstate, soOnsetArray.length);

        if(postDownSignal.length < 2){
            return false;
        }else return postDownSignal[0] < threshold && postDownSignal[0] < postDownSignal[1] && postDownSignal[0] > artifactThreshold;
    }

    float multiplyThrowTime(Double[] onsetSo, float samplingRate, float downTime) {
        float samplesDownToUp = throwMultiplier * onsetSo.length;
        float timeDownToUp = (samplesDownToUp / samplingRate) * 1000;
        float stimTimeStamp = downTime + timeDownToUp;
        return stimTimeStamp;
    }

    float correctStimTime(float stimTime, float cueDuration) {
        float deltaStimTime = cueDuration / 3;
        float output = stimTime - (2 * deltaStimTime);
        return output;
    }

    float timestampDownstate(Double[] soOnset, float currentTime, float sampleRate) {
        int numberSamples = soOnset.length - 1;
        int downSample = MathTools.argmin(soOnset);
        float timeShift = Math.round((numberSamples - downSample) * 1000 / sampleRate);
        float downstateTimestamp = currentTime - timeShift;
        return downstateTimestamp;
    }

    // NOTE: predictedSoWrite replaced with FileManager

    void masterSlowOscPrediction(Double[] uncutDelta, Double[] delta, Double[] slowDelta, int lenghtThreshold, float sampleRate, double currentTime, float cueDuration, String filename){
        // deltas[0] = onDelta, deltas[1] = onSlowDelta, deltas[2] = {idxP2n}
        Double[][] deltas = extractSlowOscillationOnset(delta, slowDelta);
        //Log.i("MasterSlowOscPredictionDeltas0", Arrays.toString(deltas[0]));
        //Log.i("MasterSlowOscPredictionDeltas1", Arrays.toString(deltas[1]));
        //Log.i("MasterSlowOscPredictionDeltas2", Arrays.toString(deltas[2]));

        Double[] onDelta = deltas[0];
        // NOTE: commented out because it is unused on source code
        //Double[] onSlDelta = deltas[1];
        Double[] idxP2n = deltas[2];
        if(idxP2n[0] == null){
            return;
        }

        // NOTE: from_end = -delta.shape[0] + idx_p2n (it gets the size in rows and columns)
        int fromEnd = (int) (-delta.length + idxP2n[0]);
        Log.i("MasterSlowOscPredictionDeltaLength", String.valueOf(delta.length));
        Log.i("MasterSlowOscPredictionFromEnd", String.valueOf(fromEnd));
        Log.i("MasterSlowOscPredictionLengthThreshold", String.valueOf(lenghtThreshold));
        // NOTE: vThreshold gets the last elements from uncutDelta, and has a len of lenghtThreshold
        Double[] vThreshold = Arrays.copyOfRange(uncutDelta, fromEnd - lenghtThreshold, fromEnd);
        double threshold = setThreshold(vThreshold);

        float downstateTime;

        if(!downstateValid(onDelta, threshold)){
            return;
        }else{
            downstateTime = timestampDownstate(onDelta, (float) currentTime, sampleRate);
        }

        stimAtStamp = multiplyThrowTime(onDelta, sampleRate, downstateTime);
        stimAtStamp = correctStimTime(stimAtStamp, cueDuration);

        String line = String.format("%s, predicted upstate at %s", downstateTime, stimAtStamp);
        Log.i("MasterSlowOscPrediction", line);
        FileManager.writeFile(appContext, filename, line);
    }
}
