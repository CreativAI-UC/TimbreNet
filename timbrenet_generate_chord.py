import os
import sys
import numpy as np
import tensorflow as tf
from scipy.io.wavfile import write
from lib.model import CVAE as Model
from lib.latent_chord import latent_chord
from lib.specgrams_helper import SpecgramsHelper


def generate_chord_from_trained_model(trained_model_path,
                                      latent_dim,
                                      sample_points,
                                      chord_saving_path):
    
    if not os.path.exists(chord_saving_path):
        os.makedirs(chord_saving_path)
    
    spec_helper = SpecgramsHelper(audio_length=64000,
                           spec_shape=(128, 1024),
                           overlap=0.75,
                           sample_rate=16000,
                           mel_downscale=1)
   
    model = Model(latent_dim)
    print('\n\nLoading Trained Model...')
    model.load_weights(trained_model_path)
    print('Success Loading Trained Model!\n')
    
    n = 1
    for sample_point in sample_points:
        chord = latent_chord(tf.constant([sample_point], dtype='float32'),model,spec_helper)
        write(chord_saving_path+'chord'+str(n)+'.wav', data = chord.audio, rate = 16000)
        print('Chord '+str(n)+' generated!')
        n += 1
    print('\n\nSUCCESS: ALL CHORDS GENERATED!    (chords are saved at '+chord_saving_path+')')
             
    
if __name__ == '__main__':
    
    #Select trained model path
    trained_model_path = './trained_models/450_piano_chords/latent_2_lr_3e-05_epoch_385_of_501'
    #trained_model_path = './trained_models/450_piano_chords/latent_8_lr_3e-05_epoch_141_of_501'
    
    #Select latent dimension 
    latent_dim = 2
    #latent_dim = 8
    
    #Select sample points
    sample_points = [[7, 8],[18,-18],[18,-7],[7,-30],[39,-10],[17,10]]
    '''
    sample_points = [[11.7 , 8.9, 12.8, 16.2,- 2.6,- 4.3,- 9.1, 21.0],
                    [- 8.0 , 9.6,-23.6, 20.0, 13.5,  8.0,-14.6,  3.1],
                    [-11.6 , 5.9,- 9.0,- 0.5,-25.4,-15.3,  3.1,  4.9],
                    [  6.3 , 3.9,  2.1,  9.1,-16.4,-13.8,- 1.8, 10.9]]
                    '''
    
    #Select path for saving chords
    chord_saving_path = './generated_chords/'
    
    generate_chord_from_trained_model(trained_model_path,
                                      latent_dim,
                                      sample_points,
                                      chord_saving_path)