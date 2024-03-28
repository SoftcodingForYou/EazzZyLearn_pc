package com.example.bmws_app;

import java.util.HashMap;
import java.util.Map;

public class Parameters {

    static String BLUETOOTH_SPP         = "00001101-0000-1000-8000-00805F9B34FB";

    // Data output parameters
    // ---------------------------------------------------------------------------------------------
    static String EEG_FILE              = "eeg.txt";
    static String STAGE_FILE            = "stage.txt";
    static String PREDICTIONS_FILE      = "predictions.txt";
    static String STIMULI_FILE          = "stimuli.txt";
    static int DATA_SAVE_INTERVAL       = 1; // scalar (s)

    static Map<String, String> SUBJECT_INFO = new HashMap<String, String>();
    static {
        SUBJECT_INFO.put("name",        "Generic");
        SUBJECT_INFO.put("age",         "Generic");
        SUBJECT_INFO.put("sex",         "Generic");
        SUBJECT_INFO.put("chosenCue",   "Generic");
    }


    // General streaming information
    // ---------------------------------------------------------------------------------------------
    static int NUM_CHANNELS             = 2;

    static Map<String, Integer> ELEC = new HashMap<String, Integer>();
    static {
        ELEC.put("Fp2", 0);
        ELEC.put("Fp1", 1);
    }

    static int IDX_ELEC                 = ELEC.get("Fp2");

    static int SAMPLE_RATE              = 200;

    // Stimulation parameters
    // ---------------------------------------------------------------------------------------------
    // Window (s) after a stim where the subject is not stimulated again
    //      - Göldi et al., 2019:       8s
    //      - Ngo et al., 2013:         2.5s
    static int LEN_REFRACTORY           = 6; // scarlar (seconds); Arbitrary, but "enough" (~Göldi)
    static String SOUND_FORMAT          = ".wav";

    // Signal structure parameters
    // ---------------------------------------------------------------------------------------------
    // All lengths are expressed in samples = duration in seconds * sampling frequency
    static int MAIN_BUFFER_LENGTH       = SAMPLE_RATE * 30; // buffer length for receiver to work with (samples = sr * 30s)

    // Buffers for processing (need to be shorter than MAIN_BUFFER_LENGTH)
    // 1. Delta buffer length
    //    This buffer can be of any length that could contain a whole slow oscillation. Must allow
    //    for thresholding buffer (this buffer will end where the delta buffer begins)
    static int DELTA_BUFFER_LENGTH      = SAMPLE_RATE * 5;
    // 2. Thresholding of downstates
    // Window in sec for downstate thresholding: "Adaptive thresholding"
    //      - Göldi et al., 2019:       No info about window size (refractory 8s)
    //      - Ngo et al., 2013:         Threshold update window of 5s
    static int THRESHOLD_BUFFER_LENGTH  = SAMPLE_RATE * 5;
    // 3. Sleep staging: 30s as in standard procedure
    static int SLEEP_BUFFER_LENGTH      = SAMPLE_RATE * 30;
    // 4. Wakening intervention: For early reaction in case of wakening; needs to be short to be
    // influenced by most recent incoming fluctuations
    static int RESPONSE_BUFFER_LENGTH   = SAMPLE_RATE * 3;


    // Slow oscillation detection (downstates) and prediction (upstates)
    // ---------------------------------------------------------------------------------------------
    static int DEFAULT_THRESHOLD        = -75; // uV
    static int NON_PHYSIOLOGICAL_THRESHOLD = -300;  // uV, below which we consider signals to be
                                                    // non-physiological of the brain

    static float THROW_MULTIPLICATION   = 1.25f;

    static int SLEEP_STAGE_INTERVAL     = 5; // scalar (s)
    static int WAKE_STAGE_INTERVAL      = 1; // scalar (s)
    static int LOCKING_LENGTH           = 120;  // scalar (s) If during the LOCKING_LENGTH time, the
                                                // majority of scores were Awake, then, no matter
                                                // the current wake staging, awake will be scored as
                                                // True. This is used in order to prevent large
                                                // movements to be interpreted as delta activity
                                                // and shift Awake to False and SWS to True


    // Signal processing
    // ---------------------------------------------------------------------------------------------
    static int FILTER_ORDER             = 3;

    static Map<String, Float[]> FREQUENCY_BANDS = new HashMap<String, Float[]>();
    static {
        FREQUENCY_BANDS.put("Delta",        new Float[]{0.5f,   4f});
        FREQUENCY_BANDS.put("SlowDelta",    new Float[]{0.5f,   2f});
        FREQUENCY_BANDS.put("Alpha",        new Float[]{8f,     12f});
        FREQUENCY_BANDS.put("Spindle",      new Float[]{12f,    16f});
        FREQUENCY_BANDS.put("Theta",        new Float[]{4f,     8f});
        FREQUENCY_BANDS.put("Beta",         new Float[]{12f,    40f});
        FREQUENCY_BANDS.put("Gamma",        new Float[]{25f,    45f});
        FREQUENCY_BANDS.put("Whole",        new Float[]{0.1f,   45f});
        FREQUENCY_BANDS.put("LineNoise",    new Float[]{49f,    51f});
    }


    // Vigilance staging parameters
    // ---------------------------------------------------------------------------------------------

    // Evaluated on n=46 EGI HGSC128 datasets using Fp1 and Scipy.signal.welch:
    // SWS staging accuracy separated at NREM1:  85.88 +/- 10.57 %
    // Wake classification separated at NREM1:   88.64 +/- 10.17 %
    // "VS" separations are crucial for method to work
    //
    static Map<String, Float> SLEEP_THRESHOLDS = new HashMap<String, Float>();
    static {
        //                   Band		        SWS if >    Separation at   Quality*
        // =========================================================================
        SLEEP_THRESHOLDS.put("DeltaVSBeta",     295f);      // NREM2        (+)
        SLEEP_THRESHOLDS.put("DeltaVSGamma",    1500f);     // NREM2        (++)
        // SLEEP_THRESHOLDS.put("AlphaVSBeta":     10f,           // NREM2        (o)
        // SLEEP_THRESHOLDS.put("AlphaVSGamma":    30,            // NREM1/2      (0)
        SLEEP_THRESHOLDS.put("SpindleVSBeta",   3f);        // NREM1        (0)
        SLEEP_THRESHOLDS.put("SpindleVSGamma",  15f);       // NREM1        (0)
        SLEEP_THRESHOLDS.put("ThetaVSBeta",     20f);       // NREM2        (+)
        SLEEP_THRESHOLDS.put("ThetaVSGamma",    100f);      // NREM1/2      (+)
    }

    static Map<String, Float> WAKE_THRESHOLDS = new HashMap<String, Float>();
    static {
        //                  Band		        Awake if >  Separation at   Quality*
        // =========================================================================
        WAKE_THRESHOLDS.put("SpindleVSTheta",   0.3f);      // Wake         (+)
        // WAKE_THRESHOLDS.put("BetaVSDelta", 0.004f);            // NREM1        (o)
        WAKE_THRESHOLDS.put("BetaVSSpindle",    0.5f);      // NREM1        (++)
        WAKE_THRESHOLDS.put("BetaVSTheta",      0.09f);     // NREM1        (++)
        WAKE_THRESHOLDS.put("Gamma",            0.5f);      // Wake         (++)
        WAKE_THRESHOLDS.put("GammaVSDelta",     0.001f);    // NREM2        (+)
        WAKE_THRESHOLDS.put("GammaVSAlpha",     0.075f);    // NREM1        (++)
        WAKE_THRESHOLDS.put("GammaVSSpindle",   0.2f);      // NREM1        (++)
        WAKE_THRESHOLDS.put("GammaVSTheta",     0.5f);      // NREM1        (++)
    }

    // * quality: (o), false negatives; (+), good; (++), perfect

    enum stagingType{
        IS_AWAKE,
        IS_SSWS
    }


    // O B S O L E T E   O R   U N U S E D   P A R A M E T E R S
    // ---------------------------------------------------------------------------------------------

    // Parameters originally used in Python version to specify connection protocol to OpenBCI
    static String IP            = "localhost";
    static int PORT             = 12345;

    // Parameter for file writing
    static String ENCODING  = "UTF-8";
}
