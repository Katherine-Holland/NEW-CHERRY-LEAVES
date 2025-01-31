import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model
from PIL import Image
from src.data_management import load_pkl_file


def plot_predictions_probabilities(pred_proba, pred_class):
    """
    Plot prediction probability results with proper logic
    to reflect the model's output.
    """
    # Define class labels and their corresponding probabilities
    class_labels = ['Uninfected', 'Infected']
    probabilities = [1 - pred_proba, pred_proba]

    # Create a DataFrame for the probabilities
    prob_per_class = pd.DataFrame({
        'Diagnostic': class_labels,
        'Probability': probabilities
    })

    # Round probabilities for better readability
    prob_per_class['Probability'] = prob_per_class['Probability'].round(3)

    # Plot the bar chart using Plotly
    fig = px.bar(
        prob_per_class,
        x='Diagnostic',
        y='Probability',
        range_y=[0, 1],
        width=600,
        height=300,
        template='seaborn',
        color='Diagnostic',
    )
    fig.update_layout(
        title=f"Prediction Confidence for {pred_class.upper()}",
        xaxis_title="Leaf Condition",
        yaxis_title="Probability",
        showlegend=False
    )
    st.plotly_chart(fig)


def resize_input_image(img, version):
    """
    Reshape image to average image size and ensure it has 3 channels (RGB).
    """
    image_shape = load_pkl_file(file_path=f"outputs/{version}/image_shape.pkl")
    # Ensure image is RGB
    if img.mode != "RGB":
        img = img.convert("RGB")
    img_resized = img.resize((image_shape[1], image_shape[0]), Image.LANCZOS)
    # Normalize pixel values
    my_image = np.expand_dims(img_resized, axis=0) / 255.0
    return my_image


def load_model_and_predict(my_image, version):
    """
    Load and perform ML prediction over live images
    """

    model = load_model(f"outputs/{version}/cherry_leaves_model.h5")

    pred_proba = model.predict(my_image)[0, 0]

    target_map = {1:"Infected", 0: "Not infected"}
    pred_class = target_map[int(pred_proba > 0.5)]

    st.write(
        f"The predictive analysis indicates the leaf is "
        f"**{pred_class.lower()}** with powdery mildew.")

    return pred_proba, pred_class
