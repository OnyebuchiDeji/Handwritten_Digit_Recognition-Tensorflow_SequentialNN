"""
    Training a Neural Network model for recognizing digits from images is kind of overkill.

    More straightforward data science techniques like K-nearest neighbour algorithms can be used
    to classify images of a certain digigt to classify them and hence identify subsequent images of a certain kind of digit.

    Similarly, a convolutional neural network can be used instread of a Sequential neural network to accomplish the same goal.

    Also, only the form of the number, not the colors, are considered

    Lastly, note that the handwritten digit images are 28x28 pixels in dimensions
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

mnist = tf.keras.datasets.mnist

"""
    Datasets are split into:
        *   Training Data
        *   Testing Data    --- Used to test the accuracy and generalisability of the model
            --- Both are labelled

    Below, the x_train data is the pixel/image data
    the y_train is the classification/label data
"""

(x_train, y_train), (x_test, y_test) = mnist.load_data()

#   Normalise Data --- scale values to be between 0->1

#   must specify axis=1 --- this normalizes the pixels' rgb(a) data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

opath = "./out/"
respath = "./res/"

model_name = "handwritten_detection.keras"

def train_model():
    if os.path.exists(opath + model_name):
        return

    """
        This is the model of the Neural Network to be trained.
        It's a basic sequential neural network.
    """
    model = tf.keras.models.Sequential()

    """
        Adding layers

        Flatten Layer: turns a 2d array of pixels into a 1D array for ease of processing
        Dense Layer: A layer where each neuron is made to be conneceted to every other neuron in
        the other layers
            --- The one used here is RELU: rectified linear unit
        Last Dense Layer: It is the output layer represented by 10 units. Each unit represents individual digits.
            --- the activation function here is `softmax`
            --- `softmax` ensures that all the outputs add up to 1; so it can be interpreted as a confidence value.
                This signals how likely the input image being interpreted represents a given digit represented by the giving output layer
                That is, each neuron of that last layer represents a digit. The neuron with the highest probability/confidence means that the
                interpreted image is most likely the digit that neuron represents.
                So if image illustrates digit, 3, the third neuron should have the highest confidence value. 
    """
    #   Flatten Layer
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

    #   Dense Layer
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))

    #   Another with smaller units because it's the output layer
    #   the 10 units represents individual digits.
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    #   Train the model
    #   epoch represents how many iterations/times the model will be trained on the same data again.
    model.fit(x_train, y_train, epochs=3)

    model.save(opath + model_name)


def test_model(shouldTest=False):
    """
        Target is low loss and high accuracy
    """
    if not shouldTest:
        return

    #   load saved model from memory
    model = tf.keras.models.load_model(opath + model_name)

    loss, accuracy = model.evaluate(x_test, y_test)

    print("Loss(%): ", loss)
    print("Accuracy(%): ", accuracy)

    return model



def interpret_digit_images():
    #   load saved model from memory
    model = tf.keras.models.load_model(opath + model_name)
    
    image_number = 0
    image_name = "image_" + str(image_number) + ".png"
    image_path = respath + image_name

    out_str = ""

    while os.path.isfile(image_path):
        try:
            image_name = "image_" + str(image_number) + ".png"
            image_path = respath + image_name
            #   the slice operator donw below is to take only one channel (the last channel)
            #   since the digit images are black and white
            img = cv2.imread(image_path)[:, :, 0]
            #   the image must itself be in a numpy array list 
            #   this is required for use in the neural network
            img = np.invert(np.array([img]))

            prediction = model.predict(img)
            inter_str = f"This digit is probably: {np.argmax(prediction)}"
            print(inter_str)
            out_str += f"for {image_name}, {inter_str}\n"
            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.show()
        except:
            print("Error!")
            break
        finally:
            image_number += 1

    with open(opath + "prediction_output.txt", "w") as wfs:
        wfs.write(out_str)


if __name__=="__main__":
    train_model()
    test_model()
    interpret_digit_images()