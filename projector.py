import os
import json
import random
import IPython
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from lib.model import TimbreNet_Model
from tensorboard.plugins import projector
from lib.specgrams_helper import SpecgramsHelper

os.environ["CUDA_VISIBLE_DEVICES"]="0"
tf.test.is_gpu_available()


RUN_ID = 'ID_2020_09_06_07_30_29'
RUN_FOLDER = './run/{}'.format(RUN_ID)

with open(os.path.join(RUN_FOLDER, 'train_params.txt')) as f:
    hparams = json.load(f)

#MODEL
TN_VAE = TimbreNet_Model(hparams['LATENT_DIM'],
                         encoder_dropout_rate=hparams['ENC_DROPOUT_RATE'],
                         encoder_use_batch_norm=hparams['ENC_BATCH_NORM'],
                         decoder_use_batch_norm=hparams['DEC_BATCH_NORM'])

TN_VAE.load_weights(os.path.join(RUN_FOLDER, 'weights/weights.h5'))
print('LOADED')

#DATASET
DATA_PATH = './datasets/pianoTriadDataset/audio_test/*'
list_test_ds = tf.data.Dataset.list_files(DATA_PATH, shuffle=True, seed=hparams['SEED'])

vect = []
meta = []
n = 0
for i in list_test_ds:
    path = str(i.numpy()).split('/')[-1]
    name = path.split('.')[0]
    data = name.split('_')
    meta.append([data[0],data[3],data[4],data[2]+data[1]])
    mel, _ = TN_VAE.pre_process(i)
    mel = tf.expand_dims(mel, 0)
    mu , logvar = TN_VAE.encoder_mu_log_var.predict(mel)
    vect.append(list(mu[0]))
    n = n+1
    if n%10 == 0:
        print(n)
    if n == 1000:
        break
print('Vect length: '+str(len(vect)))
print('Meta length: '+str(len(meta)))


def register_embedding(embedding_tensor_name, meta_data_fname, log_dir):
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_tensor_name
    embedding.metadata_path = meta_data_fname
    projector.visualize_embeddings(log_dir, config)
    
LOG_DIR=os.path.join(RUN_FOLDER, 'logs/projector/')
META_DATA_FNAME = 'meta.tsv'  # Labels will be stored here
EMBEDDINGS_TENSOR_NAME = 'embeddings'
EMBEDDINGS_FPATH = os.path.join(LOG_DIR, EMBEDDINGS_TENSOR_NAME + '.ckpt')
STEP = 0

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

register_embedding(EMBEDDINGS_TENSOR_NAME, META_DATA_FNAME, LOG_DIR)

with open(os.path.join(LOG_DIR, META_DATA_FNAME), "w", encoding='utf-8') as f:
    f.write('Instrument\tTriad\tVolume\tBaseNote\n')
    for i in meta:
        for j in i:
            f.write(str(j) + "\t")
        f.write("\n")

tensor_embeddings = tf.Variable(vect, name=EMBEDDINGS_TENSOR_NAME)
saver = tf.compat.v1.train.Saver([tensor_embeddings])  # Must pass list or dict
saver.save(sess=None, global_step=STEP, save_path=EMBEDDINGS_FPATH)