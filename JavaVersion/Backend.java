package com.example.bmws_app;
import android.content.Context;

import java.util.ArrayDeque;
import java.util.Arrays;

public class Backend {

    public Backend(Context appContext){
        this.appContext             = appContext;
        this.cueing                 = new Cueing();
        this.stage                  = new SleepWakeState(appContext);
        this.signalProcessing       = new SignalProcessing();
        this.predict                = new PredictSlowOscillation(appContext);
    }

    // Environment init
    // ---------------------------------------------------------------------------------------------
    Context appContext;


    // Pipeline-specific inits
    // ---------------------------------------------------------------------------------------------
    Cueing cueing;
    SleepWakeState stage;
    SignalProcessing signalProcessing;
    PredictSlowOscillation predict;

    void realTimeAlgorithm(ArrayDeque<Integer>[] buffer, ArrayDeque<Long> timestamps) {
        double currentTime = timestamps.getLast();

        FileManager.writeFile(appContext, Parameters.EEG_FILE, String.format("%s %s", timestamps, Arrays.toString(buffer)));

        // masterSignals indexes: [0] vWake, [1] vSleep, [2] vFilteredDelta, [3] vDelta, [4] vSlowDelta
        Double[][] masterSignals = signalProcessing.masterExtractSignal(buffer);

        // Sleep/Wake staging
        stage.masterStageWakeAndSleep(masterSignals[0], masterSignals[1], Parameters.FREQUENCY_BANDS.get("Whole"), Parameters.STAGE_FILE, currentTime);
        if(stage.isAwake || !stage.isSws){
            return;
        }

        // Slow oscillation upstate prediction
        if(predict.stimAtStamp != null){
            predict.masterSlowOscPrediction(masterSignals[2], masterSignals[3], masterSignals[4], signalProcessing.thresholdBufferLength, Parameters.SAMPLE_RATE, currentTime, cueing.cueDuration, Parameters.PREDICTIONS_FILE);
        }

        // Stimulate
        if(predict.stimAtStamp != null && currentTime >= predict.stimAtStamp && currentTime <= cueing.stimTime + cueing.lenRefractory){
            predict.stimAtStamp = null;
        }
        else if(predict.stimAtStamp != null && currentTime >= predict.stimAtStamp){
            cueing.stimTime = currentTime;
            predict.stimAtStamp = null;
            cueing.playCue();
        }
    }
}
