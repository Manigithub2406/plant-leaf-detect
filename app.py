import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import os

# All the plant species in the trained dataset
labels = ["Ulmus carpinifolia", "Acer", "Salix aurita", "Quercus", "Alnus incana", "Betula pubescens", "Salix alba 'Sericea",
          "Populus tremula", "Ulmus glabra", "Sorbus aucuparia", "Salix sinerea", "Populus", "Tilia", "Sorbus intermedia", "Fagus silvatica"]

# Function to predict the species of the plant based on the trained model and input image
def prediction(img_path, model):
    # Image pre-processing 
    img = image.load_img(img_path, target_size=(224, 224), color_mode='grayscale')
    img = image.img_to_array(img)
    img = img.reshape(-1, 224, 224, 1)
    img = img.astype('float32')
    img = img / 255.0
    # ---
    lst = model.predict(img)
    index = np.argmax(lst, axis=-1)[0]
    return index, lst[0][index]

# Applying a mask to images with a noisy background helps in more accurate classification
# This mask filters out all the green elements from the image, keeping in mind leaves are usually green
def apply_mask(img_path):
    img = cv2.imread(img_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    range1 = (36, 0, 0)
    range2 = (86, 255, 255)
    mask = cv2.inRange(hsv, range1, range2)
    result = img.copy()
    result[mask == 0] = (255, 255, 255)
    cv2.imwrite("result.jpg", result)
    return result

# Focuses mainly on the UI designs from here and calling the required functions when necessary 
def main():
    st.write("""
        ## Swedish Leaf Dataset Image Classifier 
        Feed in a leaf image to find out its species!
    """)
    image_file = st.file_uploader("Avoid choosing images with multiple leaves for better accuracy...")
    
    if st.button("Predict"):
        if image_file is not None:
            c1, c2 = st.columns([1, 5])
            with c1:
                img = Image.open(image_file)
                st.image(img, caption="Uploaded Image", width=100)
            with c2:
                # Save the uploaded file to a temporary location
                temp_file_path = "temp_image.jpg"
                with open(temp_file_path, "wb") as f:
                    f.write(image_file.getbuffer())
                st.write("Classifying...")
                model = tf.keras.models.load_model("model1.h5")
                result, acc = prediction(temp_file_path, model)
                if acc < 0.5:
                    st.warning("Low accuracy warning! Let us try using a customized mask.")
                    img2 = apply_mask(temp_file_path)
                    st.image(img2, caption="Masked Image", width=100)
                    result, acc = prediction("result.jpg", model)
                st.success(f"This leaf is from the species: {labels[result]} => Accuracy: {round(acc * 100, 2)}%")
                # Adds a google search link to the predicted species for a better user experience 
                search_fields = labels[result].split()
                search = "+".join(search_fields)
                st.write(f"[Click here to learn more about this species!](https://www.google.com/search?q={search}+leaf)")
                # Remove the temporary file
                os.remove(temp_file_path)
        else:
            st.write("Please upload an image.")

if __name__ == '__main__': 
    main()
