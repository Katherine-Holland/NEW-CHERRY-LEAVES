import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    st.write("### Project Hypothesis and Validation")

    st.success(
        f"* We suspect infected leaves will have clear marks/signs, "
        f"typically a white podery coating or shriveled leaf edges. \n\n"
        f"* An Image Montage shows that typically an infected leaf has white patches or shriveled edges. "
        f"Average Image, Variability Image and Difference between Averages studies did not reveal "
        f"any clear pattern to differentiate one from another."

    )