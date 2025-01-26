import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import time  # Added for progress bar simulation

from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import (
    load_model_and_predict,
    resize_input_image,
    plot_predictions_probabilities
)

def page_cherryleaves_detector_body():
    st.title("🍒 Cherry Leaf Powdery Mildew Detection")

    st.write("### 🔍 Project Objective")
    st.info(
        "**Objective:**\n"
        "This page allows users to upload cherry leaf images and use a machine learning model to predict "
        "whether the leaves are healthy or infected with powdery mildew.\n\n"
        "**Client Interest:**\n"
        "- The client aims to automate the detection of infected leaves to ensure product quality.\n"
    )

    st.write("### 📥 Upload Cherry Leaf Images")
    st.markdown(
        "- You can download a sample dataset of healthy and infected cherry leaves "
        "[from Kaggle here](https://www.kaggle.com/codeinstitute/cherry-leaves).\n"
        "- Upload one or more **PNG** images below to obtain predictions."
    )

    st.write("---")

    images_buffer = st.file_uploader(
        label="📤 Upload cherry leaf images (PNG format only)",
        type='png',
        accept_multiple_files=True
    )

    if images_buffer:
        df_report = pd.DataFrame([])

        # Initialize progress bar
        progress_bar = st.progress(0)
        total_images = len(images_buffer)

        for idx, image in enumerate(images_buffer):
            st.subheader(f"📄 Leaf Sample: **{image.name}**")

            # Load and display image
            img_pil = Image.open(image)
            img_array = np.array(img_pil)
            st.image(img_pil, caption=f"🖼️ Image Size: {img_array.shape[1]}px width x {img_array.shape[0]}px height")

            # Update progress
            progress_bar.progress((idx + 1) / total_images)

            # Simulating processing time
            time.sleep(1)

            # Resize and predict
            version = 'v1'
            resized_img = resize_input_image(img=img_pil, version=version)
            pred_proba, pred_class = load_model_and_predict(resized_img, version=version)

            # Display prediction results
            st.success(f"**Prediction:** The leaf is **{pred_class.upper()}** with {pred_proba*100:.2f}% confidence.")
            plot_predictions_probabilities(pred_proba, pred_class)

            # Add result to the report
            new_row = pd.DataFrame([{"Image Name": image.name, "Prediction": pred_class, "Confidence": f"{pred_proba*100:.2f}%"}])
            df_report = pd.concat([df_report, new_row], ignore_index=True)

        st.write("---")
        if not df_report.empty:
            st.success("### 📊 Prediction Summary Report")
            st.table(df_report)

            # Download report button
            st.markdown(download_dataframe_as_csv(df_report), unsafe_allow_html=True)

            st.info(
                "You can download the table above as a CSV file for further analysis."
            )

        # Complete progress bar
        progress_bar.empty()
