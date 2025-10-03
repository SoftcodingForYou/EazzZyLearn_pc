# Guidelines for sounds

## File types

* Sounds should be provided in .wav format. This is because EazzZyLearn looks up files names from the settings and automatically appends the '.wav' extension in `backend/handle_data.py` in line `self.chosen_cue  = p.SUBJECT_INFO["chosencue"] + p.SOUND_FORMAT`.

## Adding new/Modifying sounds

* Inside `frontend/settings_dialog.py`, under "Chosen cue", files that are added to this folder must be added to `self.cue_combo.addItems(["gong", "heart", "[NEW_SOUND]"])`
