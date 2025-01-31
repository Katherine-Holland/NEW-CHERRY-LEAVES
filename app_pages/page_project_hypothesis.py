import streamlit as st
import matplotlib.pyplot as plt


def page_project_hypothesis_body():
    st.title("🔬 Project Hypothesis and Validation")

    st.write("### 📋 Project Hypothesis")
    st.info(
        "Based on initial observations, our hypothesis is:\n\n"
        "**'Infected cherry leaves will exhibit visible patterns"
        " such as white powdery patches or shriveled edges, which can be"
        " detected using machine learning techniques.'**"
    )

    st.write("### 🔎 Visual Observations")
    st.success(
        "- An image montage of healthy and infected leaves"
        " suggests that **infected leaves** tend to have:\n\n"
        "  - White powdery mildew patches.\n"
        "  - Irregular shriveled edges compared to healthy leaves.\n\n"
        "- However, studies using:\n"
        "  - **Average Images** (comparing all infected vs"
        " all healthy leaves),\n"
        "  - **Variability Images**,\n"
        "  - **Difference Between Averages**,\n\n"
        "  have **not provided conclusive patterns** to reliably "
        "differentiate leaf conditions visually."
    )

    st.write("---")

    st.write("### 🧪 Hypothesis Validation Approach")
    st.markdown(
        "To validate our hypothesis, the project will follow"
        " a structured methodology that includes:\n\n"
        "1. **Data Exploration & Visualization**\n"
        "   - Conduct an in-depth study of image patterns.\n"
        "   - Generate visual comparisons such as montages.\n\n"
        "2. **Machine Learning Model Training**\n"
        "   - Train convolutional neural networks (CNNs) to"
        " learn underlying visual features.\n"
        "   - Evaluate model accuracy to detect distinguishing features.\n\n"
        "3. **Statistical Analysis**\n"
        "   - Measure model precision and recall to assess how well"
        " infected leaves are identified.\n"
        "   - Conduct statistical tests to confirm significance"
        " of findings.\n\n"
        "4. **Business Impact Evaluation**\n"
        "   - Ensure the model meets the business goal of **97% accuracy**,"
        " allowing timely interventions to prevent mildew spread.\n"
    )

    st.write("---")

    st.write("### 📊 Project Insights So Far")
    st.warning(
        "- The **current findings** suggest that visual differences are"
        " clear and the app provides an accurate identification of infected"
        " or healthy leaves.\n"
        "- Advanced deep learning models may be required to detect microscopic"
        " or color variations to possibly identify"
        " signs of early stage infection.\n"
    )

    st.write("---")

    st.write("### 🔗 Next Steps")
    st.markdown(
        "1. Perform more in-depth feature analysis"
        " to uncover hidden patterns.\n"
        "2. Evaluate model generalization with unseen"
        " data to confirm robustness.\n"
        "3. Refine image preprocessing techniques to"
        " enhance model input quality.\n"
    )

    st.info(
        "📝 The hypothesis will continue to be tested and refined"
        " as we progress through the project."
    )
