import streamlit as st
import numpy as np

from tensorflow import keras
from keras.utils import load_img, img_to_array
from matplotlib import cm
import matplotlib.pyplot as plt

from keras.applications.vgg16 import preprocess_input
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.saliency import Saliency

st.set_page_config(
    page_title="🧠 MRI Brain Tumor Detection",
)

# Load your pre-trained tumor detection model
index = {0: 'glioma', 1: 'meningioma', 2: 'normal', 3: 'adenoma'}

loaded_model = keras.models.load_model("weigts.h5")

st.title("🧠 MRI Brain Tumor Detection")

# Upload an image
uploaded_image = st.file_uploader("Upload an MRI Brain Image", type=["jpg", "png", "jpeg"])


def smoothgrad(image, model):
    score = CategoricalScore([1])
    images = np.asarray([np.array(image)])
    x = preprocess_input(images)

    saliency = Saliency(model, model_modifier=ReplaceToLinear(), clone=True)
    cam = saliency(score, x, smooth_samples=20, smooth_noise=0.2)

    return cam, images[0]


if uploaded_image is not None:
    st.image(uploaded_image, caption="Image to Process", use_column_width=True)
    origin = load_img(uploaded_image, target_size=(224, 224))

    image_to_pred = img_to_array(origin)
    image_to_pred = np.expand_dims(image_to_pred, axis=0)

    if st.button("Detect Tumor"):
        # Make predictions using your model
        result = np.argmax(loaded_model.predict(image_to_pred / 255.0), axis=1)
        st.info(f"Predicted class: **{index[result[0]]}**")

        cam, img = smoothgrad(origin, loaded_model)
        # Overlay the heatmap on the image
        fig, ax = plt.subplots(figsize=(8, 6))

        # Overlay the heatmap on the image
        heatmap = np.uint8(cm.jet(cam)[..., :3] * 255)
        ax.imshow(img)
        ax.imshow(heatmap[0], cmap='jet', alpha=0.5)
        ax.axis('off')

        st.pyplot(fig)
