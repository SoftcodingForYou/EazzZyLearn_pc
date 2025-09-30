package com.example.bmws_app;

import android.os.Handler;
import android.os.Looper;
import android.util.Log;

public class Cueing {

    int lenRefractory   = Parameters.LEN_REFRACTORY * 1000;
    // Cue duration in milliseconds
    int cueDuration     = 0;
    double stimTime     = 0;
    int nStims          = 0;

    // Called via Method Channel on Main Activity
    void setCueDuration(int duration){
        cueDuration = duration;
        Log.i("Backend", String.format("Cue is %s ms long", cueDuration));
    }

    void playCue(){
        new Handler(Looper.getMainLooper()).post(new Runnable() {
            @Override
            public void run() {
                MainActivity.cueChannel.invokeMethod("playCue","play");
            }
        });
    }
}
