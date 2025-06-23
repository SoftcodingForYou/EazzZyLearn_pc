# Recording session information (to be set for each participant)
# -------------------------------------------------------------------------

OUTPUT_DIR      = f'./EazzZyLearn_output/YYYY_mm_dd'

SUBJECT_INFO    = {
    'name':         'Generic', # string (freely choses)
    'age':          '00', # scalar (years)
    'sex':          'Female', # Male or Female
    'chosencue':    'gong',   # Sound to present during sleep
    'background':   'Meditative_Mind_P1F9MiPr2Vs_short', # Background sound to avoid silent intervals
    'cueinterval':  '1', # str (float-like: minutes)
    }


# Prepare channel information
# -------------------------------------------------------------------------

# Number of channels to be used
NUM_CHANNELS    = 4

# Select channels of interest
# Manually established list: to adapt if data structure changes
ELEC            = {}
ELEC["TP9"]     = 0
ELEC["AF7"]     = 1
ELEC["AF8"]     = 2
ELEC["TP10"]    = 3
# ELEC["Aux1"]    = 4
# ELEC["Aux2"]    = 5

IDX_ELEC        = ELEC["AF7"]


# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
#
#           +-------------------------------------------------+
#           | E N D   O F   C H A N G I N G   S E T T I N G S |
#           +-------------------------------------------------+
#
#       From here on, settings should not change between recordings
#                  and shall be adjusted only when needed.
# 
#                                   .
#                                   .
#                                   .
#
#             "With great power comes great responsability!"
#                 - Uncle Ben
#
# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


# Stimulation parameters
# -------------------------------------------------------------------------

# Window (s) after a stim where the subject is not stimulated again
LEN_REFRACTORY = 6 # scarlar (seconds); Arbitrary, but "enough" (~Göldi)
SOUND_FORMAT   = '.wav'


# # Streaming parameters
# -------------------------------------------------------------------------

SAMPLERATE      = 256
PGA             = 24
BAUDRATE        = 115200
TIMEOUT         = None


# # Streaming parameters (class Receiver)
# -------------------------------------------------------------------------

IP              = '0.0.0.0'  # Listen on all interfaces to receive from Muse-Lab
PORT            = 12345


# Buffer lengths in samples
# -------------------------------------------------------------------------

MAIN_BUFFER_LENGTH      = int(SAMPLERATE * 30) # buffer length for receiver to work with

# Buffers for processing (need to be shorter than MAIN_BUFFER_LENGTH)
# 1. Delta buffer length
#    This buffer can be of any length that could contain a whole slow 
#    oscillation. Must allow for thresholding buffer (this buffer will end 
#    where the delta buffer begins)
DELTA_BUFFER_LENGTH     = int(SAMPLERATE * 5)
# 2. Thresholding of downstates
# Window in sec for downstate thresholding: "Adaptive thresholding"
# - Göldi et al., 2019:     No info about window size (refractory 8s)
# - Ngo et al., 2013:       Threshold update window of 5s
THRESHOLD_BUFFER_LENGTH = int(SAMPLERATE * 5)
# 3. Sleep staging: 30 seconds, to be updated every 5 seconds
SLEEP_BUFFER_LENGTH     = int(SAMPLERATE * 30)
# 4. Wakening intervention: For early reactionin case of wakening
REPONSE_BUFFER_LENGTH   = int(SAMPLERATE * 3)


# Threshold parameters for slow oscillatory downstate validation
# -------------------------------------------------------------------------
DEFAULT_THRESHOLD           = -75 # (uV)
SD_MULTIPLICATOR            = 1.1   # Float, How many SDs the downstate has 
                                    # to be from mean
NON_PHYSIOLOGICAL_THRESHOLD = -300 # (uV) Below this amplitude, we do not 
                                   # consider the signal luctuation to be 
                                   # physiological


# Parameters for upstate prediction
# -------------------------------------------------------------------------
THROW_MULTIPLICATION        = 1.25


# Timings of array handling
# -------------------------------------------------------------------------

DATA_SAVE_INTERVAL      = 1 # scalar (s)
SLEEP_STAGE_INTERVAL    = 5 # scalar (s)
WAKE_STAGE_INTERVAL     = 1 # scalar (s)
LOCKING_LENGTH          = 120   # scalar (s) If during the LOCKING_LENGTH 
                                # time, the majority of scores were Awake, 
                                # then, no matter the wake staging, awake 
                                # will be scored as True. This is used in 
                                # order to prevent large movements to be 
                                # interpretated as delta activity and shift 
                                # Awake to False and SWS to True


# Muse Sleep Classifier Configuration
# -------------------------------------------------------------------------
USE_MUSE_SLEEP_CLASSIFIER = True
MUSE_METRIC_MAP = {
    "Wake": 12,
    "N1":   13,
    "N2":   14,
    "N3":   15,
    "REM":  16
}


# Filter parameters
# -------------------------------------------------------------------------

FILT_ORDER              = 3


# Frequency band parameters
# -------------------------------------------------------------------------
FREQUENCY_BANDS = {
    'Delta':    (0.5, 4),
    'SlowDelta':(0.5, 2),
    'Alpha':    (8,   12),
    'Spindle':  (12,  16),
    'Theta':    (4,   8),
    'Beta':     (12,  40),
    'Gamma':    (25,  45),
    'Whole':    (0.1, 45),
    'LineNoise':(49,  51)}


# Evaluated on n=46 EGI HGSC128 datasets using Fp1:
# SWS staging accuracy separated at NREM1:  85.88 +/- 10.57 %
# Wake classification separated at NREM1:   88.64 +/- 10.17 %
# "VS" separations are crucial for method to work
# -------------------------- SCIPY.SIGNAL.WELCH ---------------------------
#
#               Band		    SWS if >     Separation at      Quality*
# =======================================================================
SLEEP_THRESHOLDS = {
                'DeltaVSBeta':      295,        # NREM2         (+)
                'DeltaVSGamma':     1500,       # NREM2         (++)
                # 'AlphaVSBeta':      10,         # NREM2         (o)
                # 'AlphaVSGamma':     30,         # NREM1/2       (0)
                'SpindleVSBeta':    3,          # NREM1         (0)
                'SpindleVSGamma':   15,         # NREM1         (0)
                'ThetaVSBeta':      20,         # NREM2         (+)
                'ThetaVSGamma':     100         # NREM1/2       (+)
                }
#
# 		        Band                Awake if  Separation at     Quality*
# =======================================================================
WAKE_THRESHOLDS = {
                'SpindleVSTheta':   0.3,        # Wake          (+)
                # 'BetaVSDelta':      0.004,      # NREM1         (o)
                'BetaVSSpindle':    0.5,        # NREM1         (++)
                'BetaVSTheta':      0.09,       # NREM1         (++)
                'Gamma':            0.5,        # Wake          (++)
                'GammaVSDelta':     0.001,      # NREM2         (+)
                'GammaVSAlpha':     0.075,      # NREM1         (++)
                'GammaVSSpindle':   0.2,        # NREM1         (++)
                'GammaVSTheta':     0.5         # NREM1         (++)
                }

# * quality: (o), false negatives; (+), good; (++), perfect 


# Output file management
# -------------------------------------------------------------------------

MAX_BUFFERED_LINES = 1000 # int
DATA_FLUSH_INTERVAL = 30 # seconds (float)
PREDICTION_FLUSH_INTERVAL = 5 # seconds (float)
STIM_FLUSH_INTERVAL = 2 # seconds (float)
STAGE_FLUSH_INTERVAL = 5 # seconds (float)
ENCODING = 'utf-8'