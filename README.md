####    Date: 15th September, 2025


##  Handwritten Digit Recognition with a Sequential Neural Network

Uses the MNIST dataset from Tensorflow Keras to train a neural network that can recognize pixel arrangements
of digits made by hand.

This is the start of exploring Neural Network model creation with Tensorflow.

Note that the images are 28x28 pixels in dimensions, and in the setup of the model in `main.py`, this is specified in this
line of code: `model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))`. This is because the mnist dataset is trained on images
of this dimension 28x28 pixels. So if one was to predict the digits of bigger images, the images should be scaled to 28x28 pixels.

###  Observation
+   From `prediction_output.txt` in the `out` directory, you can see just how accurate it is at epochs=3.
    For better accuracy, increase the number of epochs

### Libraries Required
+  `pip install numpy opencv-python matplotlib tensorflow`


### References

**"Neural Network Python Project - Handwritten Digit Recognition", NeuralNine (2021), [Youtube]. Avaialble at: https://www.youtube.com/watch?v=bte8Er0QhDg&t=43s**


####    Date: 15th September, 2025