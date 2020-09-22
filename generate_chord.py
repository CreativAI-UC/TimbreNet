import os
import sys
import json
import numpy as np
import tensorflow as tf
from scipy.io.wavfile import write


from lib.model import TimbreNet_Model
from lib.specgrams_helper import SpecgramsHelper

def generate_chord_from_trained_model(MODEL_PATH,
                                      SAMPLE_POINTS,
                                      CHORD_SAVING_PATH):
   
    with open(os.path.join(MODEL_PATH, 'train_params.txt')) as f:
            hparams = json.load(f)
    
    TN_VAE = TimbreNet_Model(hparams['LATENT_DIM'],
                             encoder_dropout_rate=hparams['ENC_DROPOUT_RATE'],
                             encoder_use_batch_norm=hparams['ENC_BATCH_NORM'],
                             decoder_use_batch_norm=hparams['DEC_BATCH_NORM'])
    

    TN_VAE.load_weights(os.path.join(MODEL_PATH, 'weights/weights.h5'))
    print('MODEL LOADED') 
    
    if not os.path.exists(CHORD_SAVING_PATH):
        os.makedirs(CHORD_SAVING_PATH)
    
    if not os.path.exists(os.path.join(CHORD_SAVING_PATH, hparams['PARENT_TRAIN_ID'])):
        os.makedirs(os.path.join(CHORD_SAVING_PATH, hparams['PARENT_TRAIN_ID']))
    
    n = 1
    for sample_point in SAMPLE_POINTS:
        chord, _ = TN_VAE.gen_audio_and_spec(z=[sample_point])
        write(os.path.join(CHORD_SAVING_PATH, hparams['PARENT_TRAIN_ID'])+'/chord'+str(n)+'.wav', data = chord[0], rate = 16000)
        print('Chord '+str(n)+' generated!')
        n += 1
    print('\n\nSUCCESS: ALL CHORDS GENERATED! (chords saved at '+os.path.join(CHORD_SAVING_PATH,hparams['PARENT_TRAIN_ID'])+')')
             
    
if __name__ == '__main__':
    
    #Select trained model path
    MODEL_PATH = './trained_models_examples/3_latent_no_lable'
    
    #Select sample points
    SAMPLE_POINTS = [[1.48e+0,  -1.18e+0,   9.25e-2]
                    ,[3.49e-1,   3.68e-1,   9.82e-1]
                    ,[1.91e-2,   1.49e+0,  -2.60e+0]]

    
    #Select path for saving chords
    CHORD_SAVING_PATH = './generated_chords/'
    
    generate_chord_from_trained_model(MODEL_PATH,
                                      SAMPLE_POINTS,
                                      CHORD_SAVING_PATH)