import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.image import imread
from src.machine_learning.evaluate_clf import load_test_evaluation

def page_ml_performance_metrics():
    version = 'v1'

    st.title("📊 Machine Learning Performance Metrics")

    st.write("### 📂 Dataset Overview")
    st.info(
        "The dataset used for model training and evaluation consists of three distinct sets:\n"
        "- **Training Set:** Used to train the model by exposing it to a variety of images.\n"
        "- **Validation Set:** Used to fine-tune hyperparameters and prevent overfitting.\n"
        "- **Test Set:** Used to evaluate the generalization of the trained model on unseen data."
    )

    st.write("#### 🔍 Label Distribution Across Sets")
    labels_distribution = plt.imread(f"outputs/{version}/labels_distribution.png")
    st.image(labels_distribution, caption="Labels Distribution in Train, Validation, and Test Sets")

    st.write("---")

    st.write("### 📈 Model Training Performance")
    st.info(
        "The following metrics were tracked during the model training phase to assess performance:\n"
        "- **Accuracy:** Measures how often the model correctly identifies leaf health.\n"
        "- **Loss:** Indicates how well the model's predictions match the true labels."
    )

    col1, col2 = st.columns(2)
    with col1: 
        model_acc = plt.imread(f"outputs/{version}/model_training_acc.png")
        st.image(model_acc, caption="📊 Model Training Accuracy")
    with col2:
        model_loss = plt.imread(f"outputs/{version}/model_training_losses.png")
        st.image(model_loss, caption="📉 Model Training Loss")

    st.write("---")

    st.write("### 🧪 Generalized Performance on Test Set")
    st.success(
        "Evaluating the model's performance on the test set helps assess its ability to generalize to new data. "
        "The following table summarizes the key evaluation metrics."
    )

    # Load and display the model evaluation metrics
    test_evaluation_results = pd.DataFrame(
        load_test_evaluation(version), 
        index=['Loss', 'Accuracy']
    )

    st.dataframe(test_evaluation_results.style.format("{:.3f}"))

    st.write("---")

    st.write("### 📌 Key Takeaways")
    st.markdown(
        "- The model achieves an accuracy of **97%**, meeting the client's expectations.\n"
        "- The loss value indicates how well the model fits the data; lower values signify better performance.\n"
        "- Regularization techniques and data augmentation were used to minimize overfitting and improve robustness."
    )

    st.write("---")

    st.write("### 🔗 Next Steps")
    st.markdown(
        "1. Continue optimizing hyperparameters to further improve accuracy.\n"
        "2. Conduct additional tests with external datasets to assess model robustness.\n"
        "3. Explore deployment options to integrate the model into production."
    )

