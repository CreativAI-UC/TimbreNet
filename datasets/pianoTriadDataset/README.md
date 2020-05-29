# Piano Triad Dataset: 

Created by: Agustín Macaya Valladares
Date: 21-5-20

- `/audio/` folder contains 4320 examples of piano triads in .wav format recorded by a human.
- `/audio_augmented_x10/` folder contains 43200 examples of piano triads in .wav format obtained through data augmentation of `/audio/` folder.


Details:
- Sample rate: 16000 Hz
- Data type: 16-bit PCM (int16)
- File size: Each example has a file size of 128 kB (553 MB for `/audio/` folder and 5.53 GB for `/audio_augmented_x10/` folder).
- Duration: 4 seconds
- Sound: Piano.
- Chords were played by a human
- 3 seconds pressed, las second released
- 3 octaves (2,3,4).
- 12 notes per octave: Cn, Df, Dn, Ef, En, Fn, Gf, Gn, Af, An, Bf, Bn. (n is natural, f is flat).
- 4 triad types per note: Major (j), minor (n), diminshed (d), augmented (a).
- 3 volumes per triad: forte (f), metsoforte (m), piano (p).
- 10 original examples per volume + 10 data aumentation example per example = 100 examples per volume in total
- Metadata in the name of the chord. For example: "piano_3_Af_d_m_45.wav" is the 45th example of a mf (metsoforte) A flat diminished cord in the 3rd octave. 

Note:
- The audios are in 16-bit PCM (int16) data type to reduce te file size. This means that the dinamic range of values in the array is -32768 to 32768. To normilize the audios in the range -1 to 1 just divide by 32768.
