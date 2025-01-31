import streamlit as st
import matplotlib.pyplot as plt


def page_summary_body():
    st.title("🍒 Cherry Tree Mildew Detection")

    st.write("### 📄 Project Overview")
    st.markdown(
        "**Welcome to the Cherry Leaf Powdery Mildew Detection App!** \n\n"
        "This app leverages **machine learning**"
        " to analyze cherry leaf images and predict "
        "whether they are **healthy** or **infected with powdery mildew**."
    )

    st.info(
        "### 🌿 About Powdery Mildew\n"
        "- Powdery mildew is a fungal infection that primarily"
        " affects new leaf and bud growth.\n"
        "- It thrives in humid conditions and is commonly"
        " triggered by first rainfalls.\n"
        "- Early detection and treatment with fungicidal"
        " compounds are essential to prevent crop damage."
    )

    st.write("### 📊 Project Dataset")
    st.markdown(
        "- The dataset contains **4,280** carefully selected"
        " images from a total of over **27,000**, "
        "including both **healthy** and **powdery"
        " mildew-infected** cherry leaves.\n"
        "- The images are labeled to facilitate training of a"
        " machine learning model to achieve high prediction accuracy.\n"
    )

    st.markdown(
        "For additional information, please refer to the "
        "[README](https://github.com/Katherine-Holland/NEW-CHERRY-LEAVES/blob/main/README.md)."
    )

    st.success(
        "### 🎯 Business Objectives\n"
        "The project aims to achieve two key business requirements:\n\n"
        "1. **Visual Differentiation Study**\n"
        "   - Conducting a thorough analysis to visually distinguish"
        " healthy cherry leaves from those affected by powdery mildew.\n\n"
        "2. **AI-Powered Detection**\n"
        "   - Developing a machine learning model to"
        " accurately predict whether a given leaf is infected."
    )

    st.write("---")

    st.write("### 🏆 Project Goals")
    st.markdown(
        "- Achieve **97% prediction accuracy** to meet client expectations.\n"
        "- Provide an intuitive **dashboard interface** for"
        " users to analyze and predict leaf conditions.\n"
        "- Maintain **data privacy and security**,"
        " adhering to NDA agreements.\n"
    )

    st.write("### 🚀 How to Use This App")
    st.markdown(
        "1. Navigate through the provided pages to explore project insights.\n"
        "2. Upload cherry leaf images for live predictions.\n"
        "3. Download analysis reports and predictions for further review.\n"
    )

    st.info("🔍 Let's dive into the project and explore the findings!")