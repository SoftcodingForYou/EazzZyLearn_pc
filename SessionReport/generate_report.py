# Parameters
# =========================================================================
input_dir               = '/home/david/Documents/MemoRey/EazzZyLearn_output/Soledad_Acuña_Mendoza/'
run_extraction          = False # True or False for long step to be done

# Stims per hour (/X etimated hours of recordings in sleep study)
stim_range              = [
    333,
    191,
    303,
    115,
    55,
    65,
    131,
    117,
    88,
    102,
    68,
    77,
] # Needs manual updating after each user


# Prepare userland
# =========================================================================
from extract_data       import DataExtraction
from var_storage        import VariableOnDisk
from report_outputs     import GenerateOutputs
data_extraction         = DataExtraction()
var_storage             = VariableOnDisk()


# Begin report generation
# =========================================================================

if run_extraction:
    print("Extracting subject data ...")
    subject             = data_extraction.master_extraction(input_dir)
    var_storage.set('subject', subject)
    print("Extracted!")
else:
    print("Loading subject data ...")
    subject             = var_storage.get('subject')
    print("Loaded!")


generate_outputs        = GenerateOutputs(subject, input_dir, stim_range)

# Basic sleep metrics
# -------------------------------------------------------------------------
print("Extracting basic sleep metrics ...")
# generate_outputs.basic_sleep_metrics()
print("Done!")

# Stimulations
# -------------------------------------------------------------------------
print("Extracting stimulation metrics ...")
generate_outputs.stimulation_analysis()
print("Done!")