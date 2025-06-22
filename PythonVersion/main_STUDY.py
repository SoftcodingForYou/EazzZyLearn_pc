import  math
import  time
import  parameters                  as p
import  sounddevice                 as sd
import  numpy                       as np
import  numpy.matlib                as matlab
from    backend.handle_data         import HandleData
from    backend.cueing              import Cueing
from    backend.receiver            import Receiver


class Studying(Receiver):

    def __init__(self):
        
        self.HndlDt         = HandleData('Study')
        self.Cng            = Cueing()
        self.start_time     = time.perf_counter() * 1000
        self.studying       = True
        self.duration, self.soundsampling = self.HndlDt.prep_cue_info(
            self.HndlDt.cue_dir, p.SUBJECT_INFO["background"] + p.SOUND_FORMAT)
        self.bkgndsound     = self.HndlDt.prep_cue_load(
            self.HndlDt.cue_dir, p.SUBJECT_INFO["background"] + p.SOUND_FORMAT)


    def build_ambiant_sound(self):
        # Overlay tracks so that cueing track repeats every chosen interval

        cue_samples     = study.HndlDt.soundarray   # The sound array to cue with
        cue_sf          = self.HndlDt.soundsampling # Sampling frequency of 
                                                    # periodic sound
        cue_duration    = self.HndlDt.duration      # Duration of periodic sound (ms)
        s_interval      = self.Cng.studycueinterval # Interval (ms)
        back_sound      = self.bkgndsound           # Background sound array

        if cue_sf != self.soundsampling:
            raise Exception('Sound sampling frequencies must coincide')

        # We calculate how many samples we need in a whole interval
        samples_fill    = ( s_interval - cue_duration ) * cue_sf / 1000 # scalar (samples)
        samples_fill    = int(samples_fill)

        period_track    = np.hstack((np.zeros((samples_fill, )), cue_samples, ))

        s_repeat        = math.floor(back_sound.size / period_track.size + 1)
        overlay_track   = matlab.repmat(period_track, 1, s_repeat)[0]
        overlay_track   = overlay_track[0:back_sound.size]

        final_track     = back_sound + overlay_track

        return final_track, cue_sf


if __name__ == "__main__":
    
    study = Studying()
    studysound, soundsampling = study.build_ambiant_sound()

    # Launch background ambient sound
    sd.play(studysound, soundsampling, blocking = False, loop = True)

    print('Started background ambiance. Happy studying!')
    is_running = True
    while is_running:
        answer = input('Write [c] and press enter to exit program (Study session finished?): ')
        if answer == 'c':
            is_running = False
    print('You terminated your study session. Program stopped')