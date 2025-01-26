import streamlit as st
import matplotlib.pyplot as plt


def page_summary_body():
    st.title("Cherry Tree Mildew Detection")

    st.write("### Quick Project Summary\n")
    f"**This app uses machine learning to predict whether a cherry leaf is infected with powdery mildew.\n"

    st.info(
        f"**General Information**\n"
        f"* Powdery mildew is a fungal infection primarily targeting new leaf and bud growth and "
        f"ripe fruit. It is triggered by humidity such as first rainfalls\n"
        f"* A leaf is assessed in the field and if found to be infected is treated with a fungal compound\n"
        f"**Project Dataset**\n"
        f"* The available dataset contains 4280 out of +27 thousand images taken from "
        f"cherry leaves"
        f"those with powdery mildew and healthy leaves.")

    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/Katherine-Holland/CP5_mildew-detection-in-cherry-leaves/README.md).")
    

    st.success(
        f"The project has 2 business requirements:\n"
        f"* 1 - The client is interested in having a study to differentiate "
        f"an infected leaf and a healthy leaf visually.\n"
        f"* 2 - The client is interested in telling whether a given leaf is infected or not "
        )