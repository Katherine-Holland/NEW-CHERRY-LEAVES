import streamlit as st
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread
import itertools
import random

def page_cherryleaves_visualizer_body():
    st.title("🍒 Cherry Leaves Visualizer")

    st.write("### 🖼️ Visual Analysis of Cherry Leaves")
    st.info(
        "**Objective:**\n"
        "The goal of this page is to explore visual patterns and differences "
        "between healthy and powdery mildew-infected cherry leaves.\n\n"
        "**Key Insights:**\n"
        "- Visual analysis of leaf images to identify distinguishing features.\n"
        "- Assessing the effectiveness of image-based differentiation.\n"
        "- Generating montages for easy visual comparison.\n"
    )

    version = 'v1'

    # Section: Difference between average and variability images
    if st.checkbox("📊 Difference Between Average and Variability Images"):
        avg_powdery_mildew = plt.imread(f"outputs/{version}/avg_var_powdery_mildew.png")
        avg_healthy = plt.imread(f"outputs/{version}/avg_var_healthy.png")

        st.warning(
            "### Observations:\n"
            "- The average and variability images did **not** exhibit strong patterns "
            "to intuitively distinguish healthy from infected leaves.\n"
            "- However, a **subtle difference** in color pigmentation can be seen between "
            "the two categories, which might provide useful insights.\n"
        )

        st.image(avg_powdery_mildew, caption='🦠 Infected Leaf - Average and Variability')
        st.image(avg_healthy, caption='🌿 Healthy Leaf - Average and Variability')
        st.write("---")

    # Section: Differences between average infected and healthy leaves
    if st.checkbox("🔍 Differences Between Average Healthy and Infected Leaves"):
        diff_between_avgs = plt.imread(f"outputs/{version}/avg_diff.png")

        st.warning(
            "### Observations:\n"
            "- The analysis of differences between average images did not reveal "
            "distinct patterns that would allow easy differentiation.\n"
            "- More advanced feature extraction methods may be required to find meaningful visual distinctions.\n"
        )
        st.image(diff_between_avgs, caption='Difference Between Average Images')
        st.write("---")

    # Section: Image Montage
    if st.checkbox("🖼️ Generate Image Montage"): 
        st.write(
            "**How to use:**\n"
            "- Select the label category (Healthy or Infected).\n"
            "- Click on 'Create Montage' to generate a collage of images.\n"
            "- Refresh the montage to view a different set of images."
        )
        
        my_data_dir = 'inputs/cherry_leaves/cherry-leaves'
        labels = os.listdir(my_data_dir + '/validation')
        label_to_display = st.selectbox(label="Select Label Category", options=labels, index=0)

        if st.button("Create Montage"):      
            image_montage(dir_path=my_data_dir + '/validation',
                          label_to_display=label_to_display,
                          nrows=8, ncols=3, figsize=(10, 25))
        
        st.write("---")

def image_montage(dir_path, label_to_display, nrows, ncols, figsize=(15,10)):
    sns.set_style("white")
    labels = os.listdir(dir_path)

    if label_to_display in labels:
        images_list = os.listdir(f"{dir_path}/{label_to_display}")

        # Ensure the montage doesn't exceed available images
        if nrows * ncols < len(images_list):
            img_idx = random.sample(images_list, nrows * ncols)
        else:
            st.error(
                f"⚠️ Not enough images to fill the montage.\n"
                f"Available images: {len(images_list)}, requested: {nrows * ncols}.\n"
                f"Try reducing the number of rows or columns."
            )
            return
        
        # Create a figure and display images
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        plot_idx = list(itertools.product(range(nrows), range(ncols)))

        for i, idx in enumerate(plot_idx):
            img = imread(f"{dir_path}/{label_to_display}/{img_idx[i]}")
            img_shape = img.shape
            axes[idx[0], idx[1]].imshow(img)
            axes[idx[0], idx[1]].set_title(f"Size: {img_shape[1]}x{img_shape[0]} pixels")
            axes[idx[0], idx[1]].set_xticks([])
            axes[idx[0], idx[1]].set_yticks([])

        plt.tight_layout()
        st.pyplot(fig)

    else:
        st.error("❌ The selected label does not exist in the dataset.")
        st.write(f"Available options: {labels}")
