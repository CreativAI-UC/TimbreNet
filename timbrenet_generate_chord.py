import sys
import numpy as np
import tensorflow as tf
from scipy.io.wavfile import write
from lib.model import CVAE as Model
from lib.latent_chord_class import latent_chord
from lib.specgrams_helper import SpecgramsHelper



spec_helper = SpecgramsHelper(audio_length=64000,
                           spec_shape=(128, 1024),
                           overlap=0.75,
                           sample_rate=16000,
                           mel_downscale=1)


    
def main():
    ######SELECT HERE PATH OF TRAINED MODEL AND LATENT DIMENTION OF TRAINED MODEL (THEY MUST MATCH)#######
    trained_model_path = './trained_models/2019_10_25/14_54_03mel_p0_latent_2_lr_3e-05_b_1_se_1_ee_501_ep_385'
    latent_dim = 2

    ######SELECT SAMPLE POINT (NEEDS TO HAVE SAME NUMBER OF ELEMENTS AS latent_dim###########
    sample_point = [40, 25]
    if len(sample_point)!= latent_dim:
        print("\nERROR:\n\nsample_point dimention does not match latent_dim variable:")
        print("sample_point dimention:   "+str(len(sample_point)))
        print("latent_dim dimention:     "+str(latent_dim))
        sys.exit()

    model = Model(latent_dim)
    print('\n\nLoading Trained Model...')
    model.load_weights(trained_model_path)
    print("Success Loading Trained Model!\n")
    
    chord1 = latent_chord(tf.constant([sample_point], dtype='float32'),model,spec_helper)
    #IPython.display.display(IPython.display.Audio(chord1.audio, rate = 16000,normalize=False))
    
    write('test2.wav', data = chord1.audio/2, rate = 16000)
    print("\n\nSuccess, chord generated!\n\n")
    
if __name__ == '__main__':
    main()