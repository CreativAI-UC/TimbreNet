import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav

def import_audio(filename):
    audio = np.array([wav.read(filename)[1]],dtype=float)
    return audio

class latent_chord:
    
    triads     = [None,'C','Dm','Em','F','G','Am','Bdim']
    colors     = ['k', 'b','g' ,'r' ,'c','m','y' ,'0.5']
    octaves    = [None, '2','3','4']
    mkr_type   = ['o', '.','*','1']
    volumes    = [None, 'f'  ,'m'  ,'p']
    mkr_size   = [2, 500,100,20]
    instrument = ['piano','guitar']
    
    def __init__(self, latent, model, spec_helper):
        self.latent = latent
        self.model = model
        self.spec_helper = spec_helper
        self.triad = None
        self.octave = None
        self.volume = None
        self.example = None
        self.instrument = None
        
    @property
    def chord(self):
        return self.triad+self.octave
    
    @property
    def full_name(self):
        return self.instrument+'_'+self.triad+self.octave+'_'+self.volume+'_'+self.example
    
    @property
    def plt_color(self):
        return self.colors[self.triads.index(self.triad)]
    
    @property
    def plt_mkr_type(self):
        return self.mkr_type[self.octaves.index(self.octave)]
    
    @property
    def plt_mkr_size(self):
        return self.mkr_size[self.volumes.index(self.volume)]
    
    @property
    def plt_config(self):
        return self.plt_color, self.plt_mkr_type, self.plt_mkr_size
    
    @property
    def audio(self):
        spec = self.model.decode(self.latent)
        specA = spec[:,:,:,0]*13.815511 / -tf.math.reduce_min(spec[:,:,:,0])
        specB = spec[:,:,:,1]/-tf.math.reduce_min(spec[:,:,:,1])
        specA = specA/1.1
        specB = specB/1.1
        spec= tf.stack([specA,specB],axis=-1)
        audio = self.spec_helper.melspecgrams_to_waves(spec)
        audio = np.clip(audio,-0.999999,0.999999)
        return audio[0,:,0]
    
    @classmethod
    def from_file(cls, path, filename, model, spec_helper):
        cls.model = model
        cls.spec_helper = spec_helper
        if filename[-4:]=='.wav':
            audio_origin = import_audio(path+filename)[0,:]
            filename = filename[0:-4]
        else:
            audio_origin = import_audio(path+filename+'.wav')[0,:]
        mel = cls.spec_helper.waves_to_melspecgrams(audio_origin.reshape([1,64000,1]))
        latent, _ = cls.model.encode(mel)
        new_latent_chord = latent_chord(latent, model, spec_helper)
        new_latent_chord.instrument, chord_octave, new_latent_chord.volume,new_latent_chord.example = filename.split('_')
        new_latent_chord.triad = chord_octave[0:-1]
        new_latent_chord.octave = chord_octave[-1:] 
        return new_latent_chord    