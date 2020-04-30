import os
import sys
import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import CCA
from lib.model import CVAE as Model
from lib.latent_chord_v2 import latent_chord
from lib.specgrams_helper import SpecgramsHelper


def timbrenet_generate_latent_map(trained_model_path,
                                  latent_dim,
                                  dataset_path,
                                  instruments,
                                  chords,
                                  volumes,
                                  examples):
    extention = '.wav'
    spec_helper = SpecgramsHelper(audio_length=64000,
                           spec_shape=(128, 1024),
                           overlap=0.75,
                           sample_rate=16000,
                           mel_downscale=1)


    model = Model(latent_dim)
    print('\n\nLoading Trained Model...')
    model.load_weights(trained_model_path)
    print('Success Loading Trained Model!\n')

    #IMPORT AND CONVERT AUDIO TO MEL_SPECTROGRAMS
    print('\n\nImporting Dataset...')
    latent_dataset = []
    for instrument in instruments:
        for chord in chords:
            for volume in volumes:
                for example in examples:
                    latent_dataset.append(latent_chord.from_file(dataset_path,instrument+chord+volume+example+extention,model,spec_helper))

    print('Success Importing Dataset!\n')
    
    if latent_dim == 2:
        print('\n\nGenerating map for latent = 2')
    else:
        print('\n\nGenerating map for latent = '+str(latent_dim))
        print('Generating CCA Analysis...')
        X = np.zeros((len(latent_dataset),latent_dim))
        Y = np.zeros((len(latent_dataset),np.shape(latent_dataset[0].one_hot_label)[1]))

        for i in range(len(latent_dataset)):
            X[i]=latent_dataset[i].latent
            Y[i]=latent_dataset[i].one_hot_label
            
        cca = CCA(n_components=2).fit(X, Y)
        X_cca = cca.transform(X)
        print('Duccess Generating CCA Analysis!\n')
    
    #LEGEND DATA
    triads      = ['None','C','Dm','Em','F','G','Am','Bdim']
    octaves     = ['None', '2','3','4']
    colors      = [['k','k','k','k'],
                   ['b','#000077','#0000FF','#00FFFF'],
                   ['g','#005500','#00FF00','#66FF66'],
                   ['r','#AA0000','#FF0000','#FF6666'],
                   ['o','#EE6600','#FF9900','#FFCC00'],
                   ['m','#990099','#CC3399','#FF66FF'],
                   ['y','#DDDD00','#FFFF44','#FFFFCC'],
                   ['0.5','#444444','#777777','#AAAAAA']]
    volumes     = ['None', 'f'  ,'m'  ,'p']
    mkr_size    = [2, 500,100,20]
    instruments = ['None','piano','guitar']
    mkr_type    = ['o', '.','*']

    legend_mkr_list = list(itertools.product(mkr_type, list(itertools.chain.from_iterable(colors)),mkr_size))
    legend_name_list = list(itertools.product(instruments,triads, octaves, volumes))
    legend_elements = []
    legend_names = []


    #GENERATE PLOT 
    fig = plt.figure(figsize=(13,13))
    ax = fig.add_subplot(1, 1, 1)
    
    if latent_dim == 2:        
        for data in latent_dataset:    
            [[x,y]] = np.array(data.latent)
            element = ax.scatter(x, y, c=data.plt_color, marker = data.plt_mkr_type,s=data.plt_mkr_size)
            legen_data = (data.plt_mkr_type, data.plt_color , data.plt_mkr_size)
            if legen_data in legend_mkr_list:
                i = legend_mkr_list.index(legen_data)
                name = legend_name_list[i]
                legend_elements.append(element)
                legend_names.append(str(name[0])+'_'+str(name[1])+str(name[2])+'_'+str(name[3]))
                legend_mkr_list.remove(legen_data)
                legend_name_list.remove(name)
        
    else:
        for j in range(len(latent_dataset)):   
            element = ax.scatter(X_cca[j,0], X_cca[j,1], c=latent_dataset[j].plt_color, marker = latent_dataset[j].plt_mkr_type,s=latent_dataset[j].plt_mkr_size)
            legen_data = (latent_dataset[j].plt_mkr_type, latent_dataset[j].plt_color, latent_dataset[j].plt_mkr_size)
            if legen_data in legend_mkr_list:
                i = legend_mkr_list.index(legen_data)
                name = legend_name_list[i]
                legend_elements.append(element)
                legend_names.append(str(name[0])+'_'+str(name[1])+str(name[2])+'_'+str(name[3]))
                legend_mkr_list.remove(legen_data)
                legend_name_list.remove(name)
                
    plt.legend(legend_elements,legend_names,ncol=3)
    plt.title('Latent = '+str(latent_dim),fontsize=20)
    plt.grid()
    plt.show()
    
    
if __name__ == '__main__':
    #Select trained model path
    trained_model_path = './trained_models/450_piano_chords/latent_2_lr_3e-05_epoch_385_of_501'
    #trained_model_path = './trained_models/450_piano_chords/latent_8_lr_3e-05_epoch_141_of_501'
    
    #Select latent dimension 
    latent_dim = 2
    #latent_dim = 8
    
    #Select datasetr path to plot
    dataset_path = './datasets/450pianoChordDataset/audio/'
    
    #Select elements of dataset to plot
    instruments = ['piano_']
    chords = ['C2_','Dm2_','Em2_','F2_','G2_','Am2_','Bdim2_','C3_','Dm3_','Em3_','F3_','G3_','Am3_','Bdim3_','C4_']
    volumes = ['f_','m_','p_']
    examples = ['0','1','2','3','4','5','6','7','8','9']
    
    timbrenet_generate_latent_map(trained_model_path,
                                  latent_dim,
                                  dataset_path,
                                  instruments,
                                  chords,
                                  volumes,
                                  examples)