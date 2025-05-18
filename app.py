import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import imutils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.applications import VGG19

def create_model():
    # Load VGG19 without the top layers
    base_model = VGG19(weights=None, include_top=False, input_shape=(240, 240, 3))

    # Flatten the output of the base model
    x = Flatten()(base_model.output)

    # Add intermediate layers (you can adjust these layers as needed)
    x = Dense(4608, activation="relu")(x)  # Matching the Dense layers
    x = Dense(1152, activation="relu")(x)  # Intermediate Dense layer

    # Change the output layer to match the weight's shape: 2 classes for multi-class classification
    output_layer = Dense(2, activation="softmax")(x)  # Change to 2 units for multi-class

    # Create and return the model
    model = Model(inputs=base_model.input, outputs=output_layer)
    return model

def crop_brain_tumor(image, plot=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    thres = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thres = cv2.erode(thres, None, iterations=2)
    thres = cv2.dilate(thres, None, iterations=2)

    cnts = cv2.findContours(thres.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]

    if plot:
        # plt.figure()
        # plt.subplot(1, 2, 1)
        # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # plt.axis('off')
        # plt.title('Original Image')

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title('Cropped Image')
        plt.show()

    return new_image


# Load the model and weights
model = create_model()
model.load_weights("vgg19_model_03.weights.h5")  # Path to your weights file

# Title and description
st.title("Brain Tumor Detection")
st.write("Upload an MRI image, and the app will predict whether a tumor is present.")

# File uploader for MRI image
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    # st.image(uploaded_file, caption="Uploaded MRI Image", use_container_width=True)

    # Load the image
    try:
        # Open the image with PIL and convert to OpenCV format
        image = Image.open(uploaded_file).convert("RGB")
        image = np.array(image)  # Convert PIL image to numpy array (HWC format)

        # Convert RGB to BGR for OpenCV processing
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Crop the brain region
        cropped_image = crop_brain_tumor(image)

        # Convert cropped BGR image back to RGB for display
        cropped_image_rgb = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        st.image(cropped_image_rgb, caption="Processed Image")

        # Preprocess the cropped image for prediction
        cropped_image_resized = cv2.resize(cropped_image, (240, 240))  # Resize to match model input
        cropped_image_array = img_to_array(cropped_image_resized)
        cropped_image_array = np.expand_dims(cropped_image_array, axis=0)  # Add batch dimension
        cropped_image_array = preprocess_input(cropped_image_array)  # Preprocess (if needed for your model)

        # Make prediction
        prediction = model.predict(cropped_image_array)

        # The output is a vector of probabilities for each class
        class_0_prob, class_1_prob = prediction[0]

        # Interpret the result
        result = "Tumor Detected" if class_1_prob > 0.5 else "No Tumor Detected"
        confidence = class_1_prob if result == "Tumor Detected" else class_0_prob

        # Display the prediction result
        st.subheader(f"Prediction: {result}")
        st.write(f"Confidence: {confidence:.2%}")

    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")
