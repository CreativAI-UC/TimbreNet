import os
import json
import random
import IPython
import datetime
import numpy as np
from time import sleep
import tensorflow as tf
import matplotlib.pyplot as plt
from distutils.dir_util import copy_tree

from lib.model import TimbreNet_Model
from lib.specgrams_helper import SpecgramsHelper

os.environ["CUDA_VISIBLE_DEVICES"]="1"
tf.test.is_gpu_available()

hparams = {
    'LATENT_DIM'            : 32,
    'LOSS_TYPE'             : 'mse_fixed_sum',
    'R_LOSS_FACTOR'         : None,
    'ENC_DROPOUT_RATE'      : 0.0,
    'ENC_BATCH_NORM'        : True,
    'DEC_BATCH_NORM'        : False,
    
    'INITIAL_EPOCH'         : None,
    'END_EPOCH'             : None,
  
    'SEED'                  : 21,
    'LEARNING_RATE'         : 3e-5,
    'BATCH_SIZE'            : 10,
    'PRINT_EVERY_N_BATCHES' : 10,
    'NUM_TRAIN_IMAGES'      : 100,
    'NUM_TEST_IMAGES'       : 100,  
    'TRAIN_IMAGES'          : './datasets/pianoTriadDataset/audio_mini_test/*',
    'TEST_IMAGES'           : './datasets/pianoTriadDataset/audio_mini_test/*',
    
    'PARENT_TRAIN_ID'       : None, 
    'TRAIN_COMPLETED'       : False,
} 


GLOBAL_PARENT_TRAIN_ID = None                     #HERE SET THE PARENT ID, PUT NONE IF COMPLETLY NEW MODEL
R_LOSS_FACTOR_SEQUENCE = [[1,5],[0.2,5],[0.1,5]]  #[[r_loss_factor, number_epochs],[r_loss_factor, number_epochs],…]


for R_LOSS_FACTOR_PAIR in R_LOSS_FACTOR_SEQUENCE:
    
    #CREATE FOLDER FOR THIS LOOP
    time_clock = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    RUN_ID = 'ID_'+time_clock
    RUN_FOLDER = './run2/{}'.format(RUN_ID)

    if not os.path.exists(RUN_FOLDER):
        os.mkdir(RUN_FOLDER)
        os.mkdir(os.path.join(RUN_FOLDER, 'weights'))
        os.mkdir(os.path.join(RUN_FOLDER, 'logs'))
        os.mkdir(os.path.join(RUN_FOLDER + '/logs', 'scalars'))
        
        
    #IF GLOBAL TRAIN IS NEW, SET INITIAL EPOCH, END EPOCH AND R_LOSS_FACTOR    
    if GLOBAL_PARENT_TRAIN_ID == None:
        mode = 'build'
        hparams['INITIAL_EPOCH'] = 0
        hparams['END_EPOCH']     = hparams['INITIAL_EPOCH'] + R_LOSS_FACTOR_PAIR[1]
        hparams['R_LOSS_FACTOR'] = R_LOSS_FACTOR_PAIR[0]
        hparams['TRAIN_COMPLETED'] = False
        with open(os.path.join(RUN_FOLDER, 'train_params.txt'), 'w') as f:
            f.write(json.dumps(hparams))
    
    #ELSE (TRAIN STARTS FROM OTHER TRAINED MODEL) COPY THE DATA FROM PARENT TRAIN AND UPDATE  hparams
    else:
        mode = 'load'
        with open(os.path.join('./run2/{}'.format(GLOBAL_PARENT_TRAIN_ID), 'train_params.txt')) as f:
            old_hparams = json.load(f)
        hparams['INITIAL_EPOCH'] = old_hparams['END_EPOCH'] 
        hparams['END_EPOCH']     = hparams['INITIAL_EPOCH'] + R_LOSS_FACTOR_PAIR[1]
        hparams['R_LOSS_FACTOR'] = R_LOSS_FACTOR_PAIR[0]
        hparams['PARENT_TRAIN_ID'] = GLOBAL_PARENT_TRAIN_ID
        hparams['TRAIN_COMPLETED'] = False
        
        copy_tree('./run2/{}'.format(GLOBAL_PARENT_TRAIN_ID), RUN_FOLDER)
        
        with open(os.path.join(RUN_FOLDER, 'train_params.txt'), 'w') as f:
            f.write(json.dumps(hparams))

    #MODEL
    TN_VAE = TimbreNet_Model(hparams['LATENT_DIM'],
                             encoder_dropout_rate=hparams['ENC_DROPOUT_RATE'],
                             encoder_use_batch_norm=hparams['ENC_BATCH_NORM'],
                             decoder_use_batch_norm=hparams['DEC_BATCH_NORM'])
    if mode == 'build':
        TN_VAE.save(RUN_FOLDER)
        print('MODEL BUILT')
    else:
        TN_VAE.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))
        print('MODEL LOADED')   

    #DATASET
    
    #Select dataset folder
    list_train_ds = tf.data.Dataset.list_files(hparams['TRAIN_IMAGES'], shuffle=True, seed=hparams['SEED'])
    list_test_ds = tf.data.Dataset.list_files(hparams['TEST_IMAGES'], shuffle=True, seed=hparams['SEED'])

    #Apply preprocess to the dataset and batch
    audio_train_ds = list_train_ds.map(TN_VAE.pre_process).batch(hparams['BATCH_SIZE']).repeat(hparams['END_EPOCH'])
    audio_test_ds  =  list_test_ds.map(TN_VAE.pre_process).batch(1)
    
    #COMPILE MODEL
    TN_VAE.compile(hparams['LEARNING_RATE'], hparams['LOSS_TYPE'], hparams['R_LOSS_FACTOR'])
    print('RUN: '+str(RUN_ID))

    #TRAIN MODEL
    TN_VAE.train_with_generator2(     
        audio_train_ds
        , epochs = hparams['END_EPOCH']
        , steps_per_epoch = hparams['NUM_TRAIN_IMAGES'] / hparams['BATCH_SIZE']
        , run_folder = RUN_FOLDER
        , print_every_n_batches = hparams['PRINT_EVERY_N_BATCHES']
        , initial_epoch = hparams['INITIAL_EPOCH']
        , validation_data = audio_test_ds
    )

    #CONFIRM TRAIN COMPLETED
    hparams['TRAIN_COMPLETED'] = True
    with open(os.path.join(RUN_FOLDER, 'train_params.txt'), 'w') as f:
            f.write(json.dumps(hparams))
    GLOBAL_PARENT_TRAIN_ID = RUN_ID
        