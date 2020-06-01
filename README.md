# Neural-Network

Implementaion of a simple Neural Network for predicting the handwritten digits. 

There is a dataset of handwritten digits. The txt files folder contains four files; train.txt, test.txt, train-labels.txt and test-labels.txt. The dataset consists of 70,000 labeled 28x28 pixel grayscale images of hand-written digits. The dataset is split into 60,000 training images and 10,000 test images. There are 10 classes (one for each of the 10 digits). The task at hand is to train a model using the 60,000 training images and subsequently test its classification accuracy on the 10,000 test images.
The first file, train.txt, has 60,000 training samples and the second, test.txt, has 10,000 samples for you to test your code on whereas both the train-labels.txt and test-labels.txt, contains the labels for each of the images corresponding to each row of the train.txt and test.txt respectively. Each sample is a handwritten digit represented by a 28 by 28 grayscale pixel image and is a feature vector of length 784 (the input layer of the neural net contains 784 = 28 × 28 neurons). Each pixel is a value between 0 and 255, where 0 indicates white. The value of a label is the digit it represents. For instance, a label of value 8 means the sample represents the digit 8.
The format of each image in train.txt and test.txt is like the following (the square brackets included):
                            [0 255 0 0 255 198 187 0 0 ... 0 0 255]
                            
This network has three layers. One input layer, one hidden layer and one output layer. The activation function is sigmoid function. 
The output layer of the network contains 10 neurons. If the first neuron fires, i.e., has an output ≈1, then that will indicate that the network thinks the digit is a 0. If the second neuron fires then that will indicate that the network thinks the digit is a 1. And so on. A little more precisely, we number the output neurons from 0 through 9, and figure out which neuron has the highest activation value. If that neuron is, say, neuron number 6, then our network will guess that the input digit was a 6. And so on for the other output neurons.
For example: network output of [0.2, 0.4, 0.01, 0.22, 0.5, 0.8, 0.35, 0.11, 0.32, 0.1] 
means the network predicted: 5 (Highest value)


1. Chose three different learning rates: 0.01, 0.02 and 0.03
2. Assigned random weights over the links between the layers. These weights should be picked at random from an interval of [-1, +1]. You may also want to add biases to the neurons in each layer (but this is entirely up to you, all we want is how well your NN
performs!).
3. Created a neural network of size [784, 30, 10]. Since the network has three layers, it means
784 neurons in the input layer, 30 in the hidden layer and 10 in the output layer. For each training example used backpropagation algorithm to calculate the gradient estimate. It consists of the following steps:
➢ Feed forward the input to get activations of the output layer.
➢ Calculate derivatives of the cost function for that input with respect to the activations of the output layer.
➢ Calculate the errors for all the weights (and biases) of the neurons using Back propagation.
➢ Update weights (and biases).
