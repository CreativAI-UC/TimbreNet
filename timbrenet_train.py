import os
import sys
import time
import datetime
import numpy as np
import tensorflow as tf
from lib.model import CVAE as Model
from lib.latent_chord import latent_chord
from scipy.io.wavfile import read as read_wav
from lib.specgrams_helper import SpecgramsHelper

def import_audio(filename):
    audio = np.array([read_wav(filename)[1]],dtype=float)
    return audio

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi+1e-10)
    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),axis=raxis)

def compute_loss(model, x, beta):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    
    mse = tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.NONE)
    MSE = mse(x, x_logit)
    logpx_z = -tf.reduce_sum(MSE, axis=[1, 2])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + beta*(logpz - logqz_x))

def compute_apply_gradients(model, x, optimizer, beta):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, beta)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train_model(latent_dim,
                dataset_path,
                instruments,
                chords,
                volumes,
                examples,
                epochs,
                beta,
                learning_rate,
                optimizer,
                gpu = True):
    
    if gpu:
        os.environ["CUDA_VISIBLE_DEVICES"]="1"
        print(tf.test.is_gpu_available())
    
    spec_helper = SpecgramsHelper(audio_length=64000,
                                  spec_shape=(128, 1024),
                                  overlap=0.75,
                                  sample_rate=16000,
                                  mel_downscale=1)

    #IMPORT AND CONVERT AUDIO TO MEL_SPECTROGRAMS
    print('\n\nImporting Dataset...')
    
    num_examples = len(instruments)*len(chords)*len(volumes)*len(examples)
    print('Number of examples: '+str(num_examples))
    
    audio_matrix = np.zeros([num_examples, 64000, 1])
    n = 0
    for instrument in instruments:
        for chord in chords:
            for volume in volumes:
                for example in examples:
                    a = import_audio(dataset_path+instrument+chord+volume+example+'.wav')[0,:]
                    try:
                        audio_matrix[n,:,0] = a
                    except:
                        print('\nError en:  '+str(n)+'  '+str(a.shape)+'  '+dataset_path+instrument+chord+volume+example+'.wav')
                    n = n+1

    print('Success Importing Dataset!\n')
    
    print('\n\nConverting to mel spectrograms...')
    mel = spec_helper.waves_to_melspecgrams(audio_matrix)
    mel = tf.random.shuffle(mel,seed=21)
    melA = mel[0:450,:,:,0]/13.82#/13.815511 
    melB = mel[0:450,:,:,1]/1.00001 
    mel = tf.stack([melA,melB],axis=-1)
    print(mel.shape)
    print('Success converting to mel spectrograms!\n')
    
    print('\n\nPreparing train and test dataset...')
    train_melgrams = mel[0:num_examples-50,:,:,:]
    test_melgrams  = mel[num_examples-50:num_examples,:,:,:]
    TRAIN_BUF = num_examples-50
    BATCH_SIZE = 5
    TEST_BUF = 50
    train_dataset = tf.data.Dataset.from_tensor_slices(train_melgrams).shuffle(TRAIN_BUF,seed=21).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_melgrams).shuffle(TEST_BUF,seed=21).batch(BATCH_SIZE)
    print('Success preparing train and test dataset!')
    

    

    #declare model
    model = Model(latent_dim)
    model.inference_net.summary()

    print('New Model')
    description = 'mel_p0_latent_'+str(latent_dim)+'_lr_'+str(learning_rate)+'_b_'+str(beta)
    day = datetime.datetime.now().strftime("%Y_%m_%d")
    time_clock = datetime.datetime.now().strftime("%H_%M_%S")
    start_epoch = 1
    
    #Create saving variables
    train_log_dir = 'logs/gradient_tape/'+day+'/' + time_clock + description + '/train'
    test_log_dir = 'logs/gradient_tape/'+day+'/' + time_clock + description + '/test'
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    model_save = './model_weights/'+day+'/' + time_clock + description

    #Train
    best_elbo = -1e20

    for epoch in range(start_epoch, start_epoch+ epochs):
        start_time = time.time()
        train_loss = tf.keras.metrics.Mean()
        for train_x in train_dataset:
            train_loss(compute_apply_gradients(model, train_x, optimizer, beta))
        end_time = time.time()

        test_loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
              test_loss(compute_loss(model, test_x, beta))
        elbo = -test_loss.result()

        with test_summary_writer.as_default():
            tf.summary.scalar('Test ELBO', -test_loss.result(), step=epoch)
        with train_summary_writer.as_default():
            tf.summary.scalar('Train ELBO', -train_loss.result(), step=epoch)

        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch {}'.format(epoch,elbo,end_time - start_time))

        if elbo > best_elbo:
            print('Model saved:')
            best_elbo = elbo
            model.save_weights(model_save+'_se_'+str(start_epoch)+'_ee_'+str(epochs+start_epoch)+'_ep_'+str(epoch))
            model.save_weights(model_save+'_the_best')



if __name__ == '__main__':
    
    #Select latent dimension 
    latent_dim = 2
    #latent_dim = 8
    
    #Select datasetr path
    dataset_path = './datasets/450pianoChordDataset/audio/'
    
    #Select elements of dataset to plot
    instruments = ['piano_']
    chords = ['C2_','Dm2_','Em2_','F2_','G2_','Am2_','Bdim2_','C3_','Dm3_','Em3_','F3_','G3_','Am3_','Bdim3_','C4_']
    volumes = ['f_','m_','p_']
    examples = ['0','1','2','3','4','5','6','7','8','9']
    
    #Select training params
    epochs = 5
    beta = 0.2
    learning_rate = 3e-5
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    
    train_model(latent_dim,
                dataset_path,
                instruments,
                chords,
                volumes,
                examples,
                epochs,
                beta,
                learning_rate,
                optimizer,
                gpu = True)