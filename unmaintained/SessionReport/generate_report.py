# Parameters
# =========================================================================
input_dir               = r'D:\Interaxon\Codes\GitHub\EazzZyLearn_pc\EazzZyLearn_output\2025_09_04_1945\Offline_NON_Inverted'
run_extraction          = False # True or False for long step to be done

# Stims per hour (/X etimated hours of recordings in sleep study)
stim_range              = [
    296 / 2.02, # S003
    333 / 1.85, # S004
    191 / 1.11, # S005
    303 / 1.33, # S006
    138 / 2.04, # S007
    91 / 2.37, # S009
    102 / 1.05, # S013
    153 / 1.28, # S016
    165 / 2.09, # S017
    92 / 0.55, # S021
    145 / 1.15, # S027
    85 / 0.95, # S029
    77 / 2.01, # SAM 2024-01-2
] # Needs manual updating after each user


# Prepare userland
# =========================================================================
from extract_data       import DataExtraction
from var_storage        import VariableOnDisk
from report_outputs     import GenerateOutputs
data_extraction         = DataExtraction()
var_storage             = VariableOnDisk(storage_path=input_dir)


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
generate_outputs.basic_sleep_metrics()
print("   Done")

# Stimulations
# -------------------------------------------------------------------------
print("Extracting stimulation metrics ...")
generate_outputs.stimulation_analysis()
print('   Done')

print("Generating file")
generate_outputs.output_svg()
print('   Done')

print("Report generated in {}".format(generate_outputs.save_path))