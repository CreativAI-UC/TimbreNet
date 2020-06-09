from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Flatten, Dense, Reshape, BatchNormalization, Dropout, LeakyReLU, AveragePooling2D, Lambda, UpSampling2D, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

import os
import pickle
import numpy as np

class TimbreNet_Model():
    def __init__(
        self,
        latent_dim,
        input_dim                   = (128, 1024, 2),
        encoder_conv_filters        = [32,32,64,128,256,256,256,256],
        encoder_conv_kernel_size    = [1,3,3,3,3,3,3,3],
        encoder_conv_strides        = [1,1,1,1,1,1,1,1],
        encoder_average_pool_size   = [2,2,2,2,2,2,2],
        encoder_average_pool_stride = [2,2,2,2,2,2,2],
        encoder_use_dropout         = True,
        encoder_dropout_rate        = 0.2,
        encoder_use_batch_norm      = True,
        decoder_conv_t_filters      = [256,256,256,256,128,64,32,2],
        decoder_conv_t_kernel_size  = [3,3,3,3,3,3,3,1],
        decoder_conv_t_strides      = [1,1,1,1,1,1,1,1],
        decoder_up_sampling_size    = [2,2,2,2,2,2,2],
        decoder_use_dropout         = True,
        decoder_dropout_rate        = 0.2,
        decoder_use_batch_norm      = False
        ):
        
        self.name = 'TimbreNet_VAE_Model'
        
        self.input_dim                   = input_dim
        self.encoder_conv_filters        = encoder_conv_filters
        self.encoder_conv_kernel_size    = encoder_conv_kernel_size
        self.encoder_conv_strides        = encoder_conv_strides
        self.encoder_average_pool_size   = encoder_average_pool_size
        self.encoder_average_pool_stride = encoder_average_pool_stride
        self.encoder_use_dropout         = encoder_use_dropout
        self.encoder_dropout_rate        = encoder_dropout_rate
        self.encoder_use_batch_norm      = encoder_use_batch_norm
        
        self.decoder_conv_t_filters      = decoder_conv_t_filters
        self.decoder_conv_t_kernel_size  = decoder_conv_t_kernel_size
        self.decoder_conv_t_strides      = decoder_conv_t_strides
        self.decoder_up_sampling_size    = decoder_up_sampling_size
        self.decoder_use_dropout         = decoder_use_dropout
        self.decoder_dropout_rate        = decoder_dropout_rate
        self.decoder_use_batch_norm      = decoder_use_batch_norm
        
        self.latent_dim = latent_dim

        self.n_layers_encoder = len(encoder_conv_filters)
        self.n_layers_decoder = len(decoder_conv_t_filters)

        self._build()
        
    def _build(self):
        
        ### THE ENCODER
        encoder_input = Input(shape=self.input_dim, name='encoder_input')
        
        x = encoder_input
        
        conv_layer = Conv2D(
                filters = self.encoder_conv_filters[0]
                , kernel_size = self.encoder_conv_kernel_size[0]
                , strides = self.encoder_conv_strides[0]
                , padding = 'same'
                , name = 'encoder_conv_in'
                )
        
        x = conv_layer(x)
        
        for i in range(self.n_layers_encoder-1):
            conv_layer_1 = Conv2D(
                filters = self.encoder_conv_filters[i+1],
                kernel_size = self.encoder_conv_kernel_size[i+1],
                strides = self.encoder_conv_strides[i+1],
                padding = 'same',
                name = 'encoder_conv_' + str(i) + '_1'
                )
            
            conv_layer_2 = Conv2D(
                filters = self.encoder_conv_filters[i+1],
                kernel_size = self.encoder_conv_kernel_size[i+1],
                strides = self.encoder_conv_strides[i+1],
                padding = 'same',
                name = 'encoder_conv_' + str(i) + '_2'
                )
            
            avg_pool_layer = AveragePooling2D(
                pool_size=self.encoder_average_pool_size[i],
                strides=self.encoder_average_pool_stride[i],
                padding='same',
                name = 'encoder_avr_pool_' + str(i)
                )
            
            #FIRST CONV2D
            x = conv_layer_1(x)

            if self.encoder_use_batch_norm:
                x = BatchNormalization()(x)

            x = LeakyReLU()(x)

            if self.encoder_use_dropout:
                x = Dropout(rate = self.encoder_dropout_rate)(x)
            
            #SECON CONV2D
            x = conv_layer_2(x)

            if self.encoder_use_batch_norm:
                x = BatchNormalization()(x)

            x = LeakyReLU()(x)

            if self.encoder_use_dropout:
                x = Dropout(rate = self.encoder_dropout_rate)(x)
                
            #AVERAGE POOLING
            
            x = avg_pool_layer(x)
            
        shape_before_flattening = K.int_shape(x)[1:]

        x = Flatten()(x)
        self.mu = Dense(self.latent_dim, name='mu')(x)
        self.log_var = Dense(self.latent_dim, name='log_var')(x)

        self.encoder_mu_log_var = Model(encoder_input, (self.mu, self.log_var))
        
        def sampling(args):
            mu, log_var = args
            epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.)
            return mu + K.exp(log_var / 2) * epsilon
            
        encoder_output = Lambda(sampling, name='encoder_output')([self.mu, self.log_var])

        self.encoder = Model(encoder_input, encoder_output)
        
        

        ### THE DECODER

        decoder_input = Input(shape=(self.latent_dim,), name='decoder_input')
        
        x = Dense(np.prod(shape_before_flattening))(decoder_input)
        x = LeakyReLU()(x)
        x = Reshape(shape_before_flattening)(x)
        
        for i in range(self.n_layers_decoder-1):
            layer_norm = LayerNormalization(epsilon = 0.000001)
            
            up_sampling = UpSampling2D(
                size = self.decoder_up_sampling_size[i],
                interpolation = 'nearest',
                name = 'decoder_upsampling_' + str(i)
                )
            
            conv_t_layer_1 = Conv2DTranspose(
                filters = self.decoder_conv_t_filters[i],
                kernel_size = self.decoder_conv_t_kernel_size[i],
                strides = self.decoder_conv_t_strides[i],
                padding = 'same',
                name = 'decoder_conv_t_' + str(i) + '_1'
                )
            
            conv_t_layer_2 = Conv2DTranspose(
                filters = self.decoder_conv_t_filters[i],
                kernel_size = self.decoder_conv_t_kernel_size[i],
                strides = self.decoder_conv_t_strides[i],
                padding = 'same',
                name = 'decoder_conv_t_' + str(i) + '_2'
                )
            
            #UPSAMPLING 
            x = up_sampling(x)
            
            #FIRST CONV2D TRANSPOSE
            
            x = conv_t_layer_1(x)
            
            x = LeakyReLU()(x)
            
            if self.decoder_use_batch_norm:
                x = BatchNormalization()(x)
            else:
                x = layer_norm(x)
            
            #SECOND CONV2D TRANSPOSE
            
            x = conv_t_layer_2(x)
            
            x = LeakyReLU()(x)
            
            if self.decoder_use_batch_norm:
                x = BatchNormalization()(x)
            else:
                x = layer_norm(x)
        
        conv_t_layer = Conv2DTranspose(
            filters = self.decoder_conv_t_filters[-1],
            kernel_size = self.decoder_conv_t_kernel_size[-1],
            strides = self.decoder_conv_t_strides[-1],
            padding = 'same',
            activation='tanh',
            name = 'decoder_conv_t_final'
            )
        
        x = conv_t_layer(x)
            
        x = LeakyReLU()(x)

        decoder_output = x

        self.decoder = Model(decoder_input, decoder_output)
        
        ### THE FULL VAE
        model_input = encoder_input
        model_output = self.decoder(encoder_output)

        self.model = Model(model_input, model_output)
        
        
    def compile(self, learning_rate, r_loss_factor):
        self.learning_rate = learning_rate

        ### COMPILATION
        def vae_r_loss(y_true, y_pred):
            r_loss = K.mean(K.square(y_true - y_pred), axis = [1,2,3])
            return r_loss_factor * r_loss

        def vae_kl_loss(y_true, y_pred):
            kl_loss =  -0.5 * K.sum(1 + self.log_var - K.square(self.mu) - K.exp(self.log_var), axis = 1)
            return kl_loss

        def vae_loss(y_true, y_pred):
            r_loss = vae_r_loss(y_true, y_pred)
            kl_loss = vae_kl_loss(y_true, y_pred)
            return  r_loss + kl_loss

        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss = vae_loss,  metrics = [vae_r_loss, vae_kl_loss], experimental_run_tf_function=False)
        
        
    def save(self, folder):

        if not os.path.exists(folder):
            os.makedirs(folder)
            os.makedirs(os.path.join(folder, 'weights'))

        with open(os.path.join(folder, 'params.pkl'), 'wb') as f:
            pickle.dump([
                self.input_dim,
                self.encoder_conv_filters,
                self.encoder_conv_kernel_size,
                self.encoder_conv_strides,
                self.encoder_average_pool_size,
                self.encoder_average_pool_stride,
                self.encoder_use_dropout,
                self.encoder_dropout_rate,
                self.encoder_use_batch_norm,
                self.decoder_conv_t_filters,
                self.decoder_conv_t_kernel_size,
                self.decoder_conv_t_strides,
                self.decoder_up_sampling_size,
                self.decoder_use_dropout,
                self.decoder_dropout_rate,
                self.decoder_use_batch_norm, 
                self.latent_dim
                ], f)
            
            
    def load_weights(self, filepath):
        self.model.load_weights(filepath)

        
        
        

   
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        