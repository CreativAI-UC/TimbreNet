import os
import json
import random
import IPython
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from lib.model import TimbreNet_Model
from lib.specgrams_helper import SpecgramsHelper

os.environ["CUDA_VISIBLE_DEVICES"]="1"
tf.test.is_gpu_available()

'''CHOOSE PARAMETERS FROM HERE
LATENT_DIM       = [32,16,8,4,2]
LOSS_TYPE        = ['mse_fixed_mean','sigma_fixed''mse', 'mse_fixed_sum', 'sigma'] #Only 'mse_fixed_sum' works well
R_LOSS_FACTOR    = [10,100,1000,10000,100000,1,0.2]
ENC_DROPOUT_RATE = [0.0,0.1,0.2,0.4]
ENC_BATCH_NORM   = [True,False]
DEC_BATCH_NORM   = [False,True]
'''

hparams = {
    'LATENT_DIM'       : 32,
    'LOSS_TYPE'        : 'mse_fixed_sum',
    'R_LOSS_FACTOR'    : 0.1,
    'ENC_DROPOUT_RATE' : 0.0,
    'ENC_BATCH_NORM'   : True,
    'DEC_BATCH_NORM'   : False,
} 

b=a

EPOCHS = 200
BATCH_SIZE = 10
LEARNING_RATE = 3e-5
PRINT_EVERY_N_BATCHES = 10
INITIAL_EPOCH = 100
SEED = 21
NUM_TRAIN_IMAGES = 38880
NUM_TEST_IMAGES  = 4320
TRAIN_IMAGES = './datasets/pianoTriadDataset/audio_train/*'
TEST_IMAGES = './datasets/pianoTriadDataset/audio_test/*'

# run params
time_clock = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
#RUN_ID = 'ID_'+time_clock
RUN_ID = 'ID_2020_07_17_14_13_11'
RUN_FOLDER = './run/{}'.format(RUN_ID)

if not os.path.exists(RUN_FOLDER):
    os.mkdir(RUN_FOLDER)
    os.mkdir(os.path.join(RUN_FOLDER, 'weights'))
    os.mkdir(os.path.join(RUN_FOLDER, 'logs'))
    os.mkdir(os.path.join(RUN_FOLDER + '/logs', 'scalars'))
    
#mode =  'build' 
mode =  'load' 

if mode == 'build':
    with open(os.path.join(RUN_FOLDER, 'train_params.txt'), 'w') as f:
        f.write(json.dumps(hparams))

#MODEL
TN_VAE = TimbreNet_Model(hparams['LATENT_DIM'],
                         encoder_dropout_rate=hparams['ENC_DROPOUT_RATE'],
                         encoder_use_batch_norm=hparams['ENC_BATCH_NORM'],
                         decoder_use_batch_norm=hparams['DEC_BATCH_NORM'])
if mode == 'build':
    TN_VAE.save(RUN_FOLDER)
else:
    TN_VAE.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))
    print('LOADED')

#DATASET

#Select dataset folder
list_train_ds = tf.data.Dataset.list_files(TRAIN_IMAGES, shuffle=True, seed=SEED)
list_test_ds = tf.data.Dataset.list_files(TEST_IMAGES, shuffle=True, seed=SEED)

#Apply preprocess to the dataset and batch
audio_train_ds = list_train_ds.map(TN_VAE.pre_process).batch(BATCH_SIZE).repeat(EPOCHS)
audio_test_ds  =  list_test_ds.map(TN_VAE.pre_process).batch(1)

#COMPILE MODEL
TN_VAE.compile(LEARNING_RATE, hparams['LOSS_TYPE'], hparams['R_LOSS_FACTOR'])

print('RUN: '+str(RUN_ID))

TN_VAE.train_with_generator2(     
    audio_train_ds
    , epochs = EPOCHS
    , steps_per_epoch = NUM_TRAIN_IMAGES / BATCH_SIZE
    , run_folder = RUN_FOLDER
    , print_every_n_batches = PRINT_EVERY_N_BATCHES
    , initial_epoch = INITIAL_EPOCH
    , validation_data = audio_test_ds
)