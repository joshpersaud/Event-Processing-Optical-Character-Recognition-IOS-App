# CAPSTONE
Contained in this repo is our(all collaborators of the repo) senior year capstone project. We will implement an IOS app that will use a convolution neural network(CNN) to detect typed characters in an image containing a dated event and plot the event on the users calendar. 

characterCreator.py creates the training samples we will use for training of our NN.

characterExtraction.py reads the training samples created by characterCreator.py. It then processes each sample, by separating the sample from other samples in an image, it then fits the image of the sample perfectly in the user specified image size and outputs the image automatically to a folder ready to send to the NN for training.

UTF-8_Char-Num_Representation.txt is the numberical translation for the characters used in characterCreator.py.
