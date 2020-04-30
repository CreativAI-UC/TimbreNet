# TimbreNet
This project aims to create deep learning tools for musicians to work with the timbre of different sounds and instruments.

This project is in a initial developing stage.

## Datasets

- PiancoChordDatastet: Dataset with 450 piano chords audios.

## Models

- TimbreNet_PianoChordVAE: VAE for encoding piano chords in a low-dimension latent space (2D - 3D). This latent space can be used to create new sounds of piano chords or create chord sequences by moving through the latent space.

## How to generate a chord from a trained model

- Clone this repository.
- Open timbrenet_generate_chord.py
- In the "trained_model_path" variable put the file with the weights of a trained model (there are some trained models in the "trained_models" folder.
- In the "latent_dim" variable select the latent dimention of the trained model.
- In the "sample_points" variable put all the poins from where you want to sample chords. You can samplle as many chords as you want at the same time. Each point needs to have the same ammount of dimentions as the "latent_dim" variable
- In the "chord_saving_path" put the path of the folder where you want to save the chords. If the folder does not exist, the code will create it for you.
- Finally, run timbrenet_generate_chord.py

## How to generate a 2D latent map

- Clone this repository.
- Open timbrenet_generate_latent_map.py
- In the "trained_model_path" variable put the file with the weights of a trained model (there are some trained models in the "trained_models" folder.
- In the "latent_dim" variable select the latent dimention of the trained model.
- In the "dataset_path" variable select the path of the dataset you want to plot.
- In the "instruments" select the instruments of the dataset you want to plot.
- In the "chords" select the chords of the dataset you want to plot.
- In the "volumes" select the volumes of the dataset you want to plot.
- In the "examples" select the examples of the dataset you want to plot.
- Finally, run timbrenet_generate_latent_map.py
