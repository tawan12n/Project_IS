import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


st.title("Neural Network")

st.write("### DATASET")
st.write("The CIFAR-10 (Canadian Institute for Advanced Research 10) dataset is a widely used dataset for training and testing machine learning models, particularly in the field of Computer Vision. CIFAR-10 consists of small images (32x32 pixels) divided into 10 classes, each containing images of different objects. Convolutional Neural Networks (CNN) are commonly used to train models on this dataset due to their effectiveness in learning spatial hierarchies in image data.")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Select one sample image from each class
sample_images = []
sample_labels = []

for i in range(10):
    # Find the first image with the corresponding class
    image = x_train[y_train.flatten() == i][0]
    sample_images.append(image)
    sample_labels.append(class_names[i])

# Set up the Streamlit page
st.write("CIFAR-10 Sample Images")

# Display the sample images in a 2x5 grid
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
axes = axes.ravel()

for i in np.arange(10):
    axes[i].imshow(sample_images[i])
    axes[i].set_title(sample_labels[i])
    axes[i].axis('off')

# Show the plot in Streamlit
st.pyplot(fig)

st.write("### FEATURE")
st.write(" Airplane ✈️")
st.write("Automobile 🚗")
st.write("Bird 🦅")
st.write("Cat 🐱")
st.write("Deer 🦌")
st.write("Dog 🐶")
st.write("Frog 🐸")
st.write("Horse 🐎")
st.write("Ship 🚢")
st.write("Truck 🚚")

st.title(" 👾:blue[Convolutional Neural Networks (CNN)] 🤖")
st.write("""
- Convolutional Neural Network (CNN) is a deep learning model.
- CNN is most commonly applied to analyzing visual imagery.
- CNN proposed in the paper "Gradient-based learning applied to document recognition" by LeCun, Bottou, Bengio, Haffner in 1998.
""")

st.image("cnn1.png", use_container_width=True)

st.write("### Architecture of a CNN")
st.write("""
- Convolution layer (with activation function)
- Pooling layer
- Flatten layer
- Fully Connected layer (ANN)
""")

st.image("cnn2.png",caption="Convolution layer and Pooling layer can be combined in many different ways", use_container_width=True)

st.write("### Convolution Layer", divider="red")
st.write("#### Used for preserve spatial structure")
st.write("#### **Example configuration in Tensorflow:**")
code = '''
Conv2D(
    filters=32, # How many filters we will learn
    kernel_size=(3, 3), # Size of filters
    strides=(1, 1), # How the feature map "steps" across the image
    activation='relu', # Rectified Linear Unit Activation Function
    input_shape=(28, 28, 1) # The expected input shape for this layer
)
 '''
st.code(code, language="python")
st.image("cnn3.png", use_container_width=True)

st.write("### Convolutional Method")
st.image("cnn4.png", use_container_width=True) ##### 8

st.write("### Convolution Layer")
st.image("cnn5.png", use_container_width=True) ##### 9
st.write("#### if we had 6 filters, we’ll get 6 separate feature maps")
st.image("cnn6.png", use_container_width=True)  ##### 10
st.image("cnn11.png", use_container_width=True)  ##### 14

st.write("### Filter & Striding")
st.image("cnn7.png",caption="Filter", use_container_width=True) ##### 11
st.write("#### More filters mean more edge or feature detection from original image")

st.image("cnn8.png",caption="Stride", use_container_width=True) ##### 11

st.write("### Padding")
st.image("cnn9.png", use_container_width=True)  ##### 12

st.write("### Convolution Output Size")

st.image("cnn10.png", use_container_width=True)  ##### 13
st.write("Output size: (N - F) / stride + 1")
st.write("e.g. N = 7, F = 3")
# Bullet points with examples
st.write("""
- stride 1 => (7 - 3)/1 + 1 = 5
- stride 2 => (7 - 3)/2 + 1 = 3
- stride 3 => (7 - 3)/3 + 1 = 2.33 :(
""")

st.write("### Activation function")
st.image("cnn12.png",caption="“Actual Result of Convolution Layer”", use_container_width=True) ##### 15

st.write("### Pooling Layer")
st.write("makes the feature map (image) smaller and more manageable")
st.write("Output size:")
st.write("""
- Output size: 
- (N - F) / stride + 1 ---> (9 - 3)/3 + 1 = 3
""")
st.image("cnn13.png",caption="keras.layers.MaxPooling2D( (3, 3) )", use_container_width=True) ##### 16

st.write(""" - Output of max-pooling """)
st.image("cnn15.png", use_container_width=True)  ##### 17
st.write(""" - Output of average-pooling """)
st.image("cnn16.png", use_container_width=True)  ##### 17
st.image("cnn14.png",caption="Output size: (N - F) / stride + 1", use_container_width=True) ##### 17

st.write("### Max Pooling")
st.image("cnn17.png", use_container_width=True)  ##### 18

st.write("### Flatten Layer")
st.image("cnn18.png", use_container_width=True)  ##### 19

st.write("### Fully Connected Layer (FC Layer)")
st.write(" Fully Connected Layer = Ordinary Neural Networks")
st.image("cnn19.png", use_container_width=True)  ##### 20A


st.title(" :red[Develop CNN Model]")

st.write("### Importing Necessary Libraries")
code = '''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_splitr
 '''
st.code(code, language="python")
st.write("""
- pandas: A library for data manipulation and analysis.
- numpy: Used for numerical operations, often for arrays and matrices.
- matplotlib.pyplot: A plotting library for creating visualizations such as graphs and charts.
- tensorflow: An open-source deep learning library for building neural networks.
- keras: A high-level API for TensorFlow, which simplifies the process of building and training neural networks.
- train_test_split: From sklearn, used to split datasets into training and testing sets (though not used directly in the provided code).
""")

st.write("### Loading the CIFAR-10 Dataset")
code = '''
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
 '''
st.code(code, language="python")
st.write("""
- keras.datasets.cifar10.load_data(): Loads the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 different classes. The data is split into training and test sets
- x_train: The image data for the training set (50,000 images).
- y_train: The corresponding labels for the training images.
- x_test: The image data for the test set (10,000 images).
- y_test: The corresponding labels for the test images.
""")

st.write("### Normalizing the Images")
code = '''
x_train, x_test = x_train / 255.0, x_test / 255.0
 '''
st.code(code, language="python")
st.write("""
- Normalization: The pixel values in images range from 0 to 255. Dividing by 255 scales the values to a range of 0 to 1. This is a common practice to help speed up the training and improve the convergence of neural networks.
""")

st.write("### Defining the Model Architecture")
code = '''
model = keras.Sequential([
    keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

 '''
st.code(code, language="python")
st.write("""
- keras.Sequential(): This is used to define a linear stack of layers in a neural network.
- Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)): A 2D convolutional layer
- MaxPooling2D((2, 2)): A pooling layer to reduce the spatial dimensions (downsample) by taking the maximum value in a 2x2 window. This reduces the computation for the next layers.
- Flatten(): Flattens the 2D output from the previous layers into a 1D array to feed into the fully connected layers.
- Dense(64, activation='relu'): A fully connected layer with 64 neurons and ReLU activation.
- Dense(10, activation='softmax'): The final output layer with 10 neurons (one for each class in CIFAR-10) and Softmax activation to output probabilities for each class.
""")

st.write("### Compiling the Model")
code = '''
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
 '''
st.code(code, language="python")
st.write("""
- optimizer='adam': The Adam optimizer is used, which is an adaptive learning rate optimization algorithm.
- loss='sparse_categorical_crossentropy': The loss function used for multi-class classification where the labels are integers. It calculates how well the model's predictions match the true labels.
- metrics=['accuracy']: Accuracy is used as the evaluation metric during training and testing.

""")

st.write("### Training the Model")
code = '''
model.fit(x_train, y_train, epochs=50, batch_size=32, validation_split=0.2)
 '''
st.code(code, language="python")
st.write("""
- model.fit(): Trains the model on the training data
""")

st.image("cnn20.png", use_container_width=True)  ##### 20A

st.write("### Layers")
st.image("cnn21.png", use_container_width=True)  ##### 20A