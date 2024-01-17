# Chordonomicon
Chordonomicon: A Dataset of 666,000 Chord Progressions

Chordonomicon is a very large scale dataset featuring the symbolic representation of more than 666,000 contemporary music compositions through the use of music chords and chord progressions. We offer metadata for details such as genre, sub-genre, and release date. Additionally, we include structural information related to different parts of the music piece. Chord progressions are also represented as graphs, poised to make a noteworthy contribution to the graph machine-learning community by providing a more expansive and diversified resource. We offer three Python scripts: one for transposing chords into all tonalities (for data augmentation purposes), another for converting chords into their corresponding notes (e.g., A:7 → ['la','do\#,'mi','sol']), and a third script that generates a binary 12-semitone list representation for each chord, commencing with the note C (e.g., C:maj7 → [1,0,0,0,1,0,0,1,0,0,0,1]) (all scripts are in convert_to_mappings.ipynb).

The full dataset can be downloaded from here:
https://drive.google.com/file/d/1GUkFqV6L5XWy5rnolMC_elsMKwikDd5I/view?usp=sharing
