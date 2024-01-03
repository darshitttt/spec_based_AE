# spec_based_AE

This repo has the code that:
1. Converts an audio file to logMel representation
2. Trains an autoencoder that can regenerate the same representation (with a hidden state of dim [16,10,81])
3. Trains another decoder to detect and recognise sound events using NIGENs Dataset
   1. The decoder classifier can right now classify between 14 different classes.


### Future Aim
To investigate if this representation is good enough to train a general sound event classifier. 
- We can also think of using the classifier to solve speech-related classification task such as Speaker Diarisation, Speaker gender recognition, speaker verification, etc. 
- An idea also could be somehow integrate context into it, so that we can have a more accurate and crisp event classes to detect and recognise from.


PS: I am also trying to rewrite the code in a cleaner manner. 