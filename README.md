# Chordonomicon
Chordonomicon: A Dataset of 666,000 Chord Progressions

Chordonomicon is a very large scale dataset featuring the symbolic representation of more than 666,000 contemporary music compositions through the use of music chords and chord progressions. We offer metadata for details such as genre, sub-genre, and release date. Additionally, we include structural information related to different parts of the music piece. Chord progressions are also represented as graphs, poised to make a noteworthy contribution to the graph machine-learning community by providing a more expansive and diversified resource. We offer three Python scripts: one for transposing chords into all tonalities (for data augmentation purposes), another for converting chords into their corresponding notes (e.g., A:7 → ['la','do\#,'mi','sol']), and a third script that generates a binary 12-semitone list representation for each chord, commencing with the note C (e.g., C:maj7 → [1,0,0,0,1,0,0,1,0,0,0,1]) (all scripts are in convert_to_mappings.ipynb).

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
