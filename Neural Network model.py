import streamlit as st
from tensorflow import keras
import numpy as np
from PIL import Image

# Load model
model = keras.models.load_model('cifar10_cnn_model.h5')

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Define class names (CIFAR-10 class labels)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

st.title('CIFAR-10 Image Classifier')
st.write(f"Supported classes: {', '.join(class_names)}")

# Image preprocessing function
def preprocess_image(image):
    image = image.resize((32, 32))
    image = np.array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image

# Allow user to enter a CIFAR-10 class name
user_input = st.text_input("Enter CIFAR-10 class name:")

if user_input.lower() in class_names:
    class_index = class_names.index(user_input.lower())
    class_image = x_train[y_train.flatten() == class_index][0]
    image = Image.fromarray(class_image)

    st.image(image, caption=f"Sample Image for {user_input.capitalize()}", use_column_width=True)

    # Process and predict the class for the sample image
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)[0]
    sorted_indices = np.argsort(predictions)[::-1]  # Sort predictions in descending order
    
    top_predictions = [(class_names[i], predictions[i] * 100) for i in sorted_indices[:3]]
    
    st.write("Prediction probabilities:")
    for label, confidence in top_predictions:
        st.write(f"{label}: {confidence:.2f}%")

    predicted_class, predicted_confidence = top_predictions[0]
    st.write(f"The model predicts this is a {predicted_class} with {predicted_confidence:.2f}% confidence.")
elif user_input:
    st.write("Please enter a valid CIFAR-10 class name.")

# Allow user to upload an image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Process and predict the class for the uploaded image
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)[0]
    sorted_indices = np.argsort(predictions)[::-1]  # Sort predictions in descending order
    
    top_predictions = [(class_names[i], predictions[i] * 100) for i in sorted_indices[:3]]
    
    st.write("Prediction probabilities:")
    for label, confidence in top_predictions:
        st.write(f"{label}: {confidence:.2f}%")

    predicted_class, predicted_confidence = top_predictions[0]
    st.write(f"The model predicts this is a {predicted_class} with {predicted_confidence:.2f}% confidence.")
else:
    st.write("Please upload an image to classify.")