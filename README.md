# The-Perceptron-Hvass-Labs
A single layer feed forward neural network to classify the popular MNIST handwritten digits

### Objective
+ To classify the MNIST handwritten digits dataset.
+ To evaluate the model's Accuracy.

### Dependencies
+ `tensorflow` for building the Neural Network.
+ `numpy` for scientific computing and using numpy's `array`.
+ `matplotlib` to visulaise the accurate and inaccurate predictions
+ `sklearn` to get the confusion matrix of the sensitivity and subjectivity.
+ `tqdm` for progress bar while the network is training and to know how long it'll take to complete.

### Method
+ Load in the dataset using the `read_data_sets()` function in tensorflow.
+ `one_hot` is set to `True` for computational reasons.
+ Building the model is divided into two parts: – Building the computational graph and – Running the graph.
##### Building `tensorflow`'s computational graph
+ Build the tensorflow's computational graph which defines all the computational operations (called ops in tensorflow's grammar).
+ Two placeholder variables are created; `X` and `y` which would be feed into the network later on during training.
+ The input variable `X` is propagated forward through the network to determine the output.
+ The error between the model's prediction and ground truth is determined using tensorflow's `softmax_cross_entropy_with_logits()` function.
+ The cost function is gotten by averaging the cross entropy value using the tensorflow's helper function `reduce_mean()`.
+ `GradientDescentOptimizer().minimize()` is used to minimize the cost function by updating the `weights` and `biases` using Chain Rule's partial derivative.
+ The above method is called _Back Propagation_ in Deep Learning which is a popular method for training a neural network.
+ The overall accuracy of the model is also determined by evaluating the output which is predicted correctly using tensorflow's `equal()` and `argmax()` function.
+ The accuracy is printed out to the console to let us know how accuracte the model was
##### Running `tensorflow`'s computational graph
+ In order to run the tensorflow's computational graph, we define a `Session()` variable which encapsulate the above operations into a default graph.
+ Using the tensorflow's `global_variables_initializer()`, we initialize all the variables created in the above computational graph.
+ The network is trained for a set number of iterations.
+ The accuracy  is being run through the `Session()` variable created.
+ More visualizations is done to evaluated visually the model's accuracy (correct predictions) and flaws (incorrect predictions).
+ The confusion matrix is also printed to the console to gain more insights on the model's behaviour

### Result
+ 93.7% Accuracy on the MNIST dataset (which is considered laughable infact this is very poor).
+ Demonstration of how _Forward Propagation_ and _Backward Propagation_ works.
+ Visualizing the so called `weights` of the neural network.
+ Visualizing the confusion matrix.
