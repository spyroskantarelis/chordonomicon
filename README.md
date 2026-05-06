# Chordonomicon
Chordonomicon: A Dataset of 666,000 Chord Progressions

Chordonomicon is a very large scale dataset containing containing over 666,000 song-level symbolic chord progressions, annotated with structural parts (verse, chorus, bridge, etc.), genre, and release date, created by scraping various sources of user-generated progressions and associated metadata, showing strong similarity to well-established prior datasets. Beyond the dataset itself, we propose a reproducible benchmark suite for next chord prediction, evaluating three sequence modeling architectures (RNN, GRU, LSTM) across multiple context window sizes and data scales under strict exact-match evaluation. Our experiments reveal that structural part annotations consistently improve prediction performance. Chordonomicon is released as an open benchmark, providing split methodology, baselines, and evaluation protocols to enable fair and reproducible comparison for future work on chord prediction, classification, generation, and beyond. All code and information for replicating our next chord prediction benchmark is in Chord_Prediction branch.

Using convert_mirex.ipynb you can convert the chord progressions into Harte syntax.

We  offer three additional Python scripts: one for transposing chords into all tonalities (for data augmentation purposes), another for converting chords into their corresponding notes (e.g., A:7 → ['la','do\#,'mi','sol']), and a third script that generates a binary 12-semitone list representation for each chord, commencing with the note C (e.g., C:maj7 → [1,0,0,0,1,0,0,1,0,0,0,1]) (all scripts are in convert_to_mappings.ipynb).

The full updated dataset (as of 12/3/2024) can be downloaded from here:
https://huggingface.co/datasets/ailsntua/Chordonomicon

For a detailed description of the Chordonomicon Dataset, please see our paper on arXiv [https://doi.org/10.48550/arXiv.2410.22046]. If you use this dataset, kindly cite the paper to acknowledge the work.

### Citation
> @article{kantarelis2024chordonomicon,
  title={CHORDONOMICON: A Dataset of 666,000 Songs and their Chord Progressions},
  author={Kantarelis, Spyridon and Thomas, Konstantinos and Lyberatos, Vassilis and Dervakos, Edmund and Stamou, Giorgos},
  journal={arXiv preprint arXiv:2410.22046},
  year={2024}
}
