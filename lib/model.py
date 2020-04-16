import tensorflow as tf

class CVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(128, 1024, 2)),
                
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=1, strides=1, padding='same'),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=128, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.Conv2D(
                    filters=256, kernel_size=3, strides=1, padding='same',activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.AveragePooling2D(
                    pool_size=(2, 2),strides=2,padding='same'),
                
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(latent_dim + latent_dim),
          ]
        )
        
        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=1*8*256),
                tf.keras.layers.Reshape(target_shape=(1, 8, 256)),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=256, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=128, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=128, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=64, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.UpSampling2D(
                    size = 2, interpolation='nearest'),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=(2,16), strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                tf.keras.layers.Conv2DTranspose(
                    filters=32, kernel_size=3, strides=1, padding="SAME", activation=tf.keras.layers.LeakyReLU()),
                tf.keras.layers.LayerNormalization(
                epsilon=0.000001),
                
                tf.keras.layers.Conv2DTranspose(
                    filters=2, kernel_size=1, strides=1, padding="SAME", activation='tanh'),
            
            ]
        )
        
    def sample(self, eps=None, apply_tanh=True):
        if eps is None:
            eps = tf.random.normal(shape=(2, self.latent_dim))
        return self.decode(eps, apply_tanh)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_tanh=False):
        logits = self.generative_net(z)
        if apply_tanh:
            probs = tf.math.multiply(tf.math.tanh(logits),0.999999)
            return probs
        else:
            probs = tf.math.multiply(logits,0.999999)
        return logits