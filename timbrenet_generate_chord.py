import os
import logging
import sys
import numpy as np
import tensorflow as tf
from scipy.io.wavfile import write
# import pydub
from .lib.latent_chord import latent_chord
import time
from pydub import AudioSegment
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

def generate_chord_from_trained_model(trained_model_path,
                                      latent_dim,
                                      sample_point,
                                      chord_saving_path,model,spec_helper):
    
    if not os.path.exists(chord_saving_path):
        os.makedirs(chord_saving_path)

    file_time = str(time.time())
    filename = chord_saving_path+file_time+'.wav'

    chord = latent_chord(tf.constant([sample_point], dtype='float32'),model,spec_helper)
    write(filename, 16000, chord.audio)

    sound = AudioSegment.from_wav(filename)
    filename = chord_saving_path+file_time+'.mp3'
    sound.export(filename, format='mp3')

    return filename
             
    
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
