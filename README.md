# Cherry Leaf Powdery Mildew Detection

## 📄 **Project Overview**
This project aims to assist Farmy & Foods, an agricultural company, in automating the detection of powdery mildew in cherry leaves. Powdery mildew is a fungal disease that affects plant health and fruit quality, posing a significant challenge for the company. The manual detection process is time-consuming and inefficient, making scalability a challenge. This project utilizes machine learning to predict whether a cherry leaf is healthy or infected with powdery mildew, using images provided by the client.

The project delivers a **Streamlit dashboard** that fulfills the client's business requirements and provides a scalable, efficient solution to detect mildew.

---

## 🎯 **Business Requirements**
1. **Visual Differentiation Study**  
   Conduct a study to visually differentiate cherry leaves that are healthy from those infected with powdery mildew. This study includes:
   - Average images and variability images for both classes (healthy and infected).
   - Difference between the average healthy and infected leaves.
   - Image montages showcasing both classes.

2. **Predictive Capability**  
   Build a machine learning system capable of predicting whether a cherry leaf is healthy or infected based on an input image, achieving a minimum accuracy of 97%.

3. **Dashboard Development**  
   Develop a user-friendly dashboard where:
   - Users can upload images for live prediction.
   - The dashboard displays prediction results, probabilities, and related visualizations.
   - Summary reports can be downloaded for further analysis.

---

## 📂 **Dataset**
The dataset, sourced from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves), consists of **4,280 images of cherry leaves**:
- **Healthy**: Cherry leaves with no signs of infection.
- **Infected**: Cherry leaves with visible powdery mildew.

### Dataset Details:
- **Train, Validation, and Test Splits**:  
  The dataset is split into training, validation, and test sets for effective model training and evaluation.
- **Preprocessing**: Images were resized and augmented to enhance model performance. Grayscale conversion was tested to optimize runtime without compromising accuracy.

---

## 🧪 **Project Hypothesis**
The hypothesis for the project was that infected cherry leaves would exhibit:
1. **White powdery mildew patches.**
2. **Shriveled or irregular edges.**

### Validation Approach
The hypothesis was tested using:
- Visual analysis through montages, average images, variability images, and difference studies.
- Training a Convolutional Neural Network (CNN) to classify leaves as healthy or infected.
- Evaluating the model's accuracy and loss on training, validation, and test sets.

---

## 🛠️ **Machine Learning Pipeline**
### Key Steps:
1. **Data Preprocessing:**
   - Resize images to manageable dimensions (256x256 pixels).
   - Experiment with grayscale conversion to optimize runtime and to test accuracy.

2. **Model Architecture:**
   - Build using a **Convolutional Neural Network (CNN)**.
   - Incorporate **dropout layers** and **early stopping** to prevent overfitting.
   - The **grayscale model** will be trialled to see if it improves runtime and or accuracy.

3. **Evaluation Metrics:**
   - Accuracy and loss on training, validation, and test sets.

4. **Deployment:**
   - Integrate the model into a Streamlit dashboard for real-time predictions.

---

## 📊 **Dashboard Features**
The dashboard is structured into the following pages:

### **Page 1: Quick Project Summary**
- Overview of the project, dataset, and business requirements.
- Explanation of powdery mildew and its impact.

### **Page 2: Leaf Visualizer**
- Visual analysis:
  - Image montages for both healthy and infected leaves.
  - Average and variability images for each class.
  - Differences between the average healthy and infected images.
  - Uses checkboxes to select relevant information.

### **Page 3: Powdery Mildew Detection**
- File uploader for users to upload images for predictions.
- Displays uploaded images, prediction results, and probabilities.
- Downloadable summary report of predictions.

### **Page 4: Project Hypothesis**
- Outlines project hypothesis and validation process.
- Highlights current findings from visual analysis and model training.

### **Page 5: Model Performance Metrics**
- Both colour and greyscale models were tested for:
- Training accuracy and loss plots.
- Test set performance metrics.
- Visual charts to show data results were used to show accuracy and loss.

---
## Epics and User Stories

## 🌱 Epic 1: Data Collection and Preparation  
*As a data scientist, I need to collect and preprocess data so that I can train an accurate model.*

### **User Stories:**
- **As a data scientist,** I want to gather a dataset of cherry leaves (healthy and infected) from Kaggle so that I have labeled images for model training.  
- **As a data engineer,** I want to analyze the dataset to understand its distribution and balance across different classes so that I can ensure a fair model training process.  
- **As a data scientist,** I want to preprocess the dataset by resizing images and normalizing pixel values so that they are compatible with my machine learning model.  
- **As a data analyst,** I want to store my cleaned dataset securely under an NDA to protect client confidentiality.  

---

## 📊 Epic 2: Data Visualization and Exploratory Analysis  
*As a data scientist, I need to explore the dataset visually so that I can understand the key patterns and insights.*  

### **User Stories:**
- **As a researcher,** I want to generate an average and variability image for healthy and infected leaves so that I can identify key visual differences.  
- **As a data analyst,** I want to create an image montage to display a range of healthy and infected leaves so that users can visually compare them.  
- **As a scientist,** I want to generate histograms and bar charts showing the dataset distribution across training, validation, and test sets so that I can ensure data is split correctly.  
- **As a researcher,** I want to analyze grayscale vs. color images to determine if grayscale preprocessing improves classification accuracy and or speed of analysis.  

---

## 🧠 Epic 3: Model Development and Optimization  
*As a machine learning engineer, I need to develop and train a neural network to classify cherry leaves as healthy or infected.*  

### **User Stories:**
- **As a machine learning engineer,** I want to define a convolutional neural network (CNN) architecture so that it can accurately classify cherry leaves.  
- **As a data scientist,** I want to experiment with greyscale to find the best trade-off between model accuracy and file size.  
- **As a model developer,** I want to train the model using early stopping to prevent overfitting and optimize its generalization.  
- **As a researcher,** I want to compare model performance using accuracy, precision, recall, and F1-score to validate its effectiveness.  

---

## 🖥️ Epic 4: Dashboard Development  
*As a product owner, I need a user-friendly dashboard so that clients can easily upload images and receive predictions.*  

### **User Stories:**
- **As a business user,** I want to see a project summary page so that I understand the purpose and scope of the solution.  
- **As a researcher,** I want to see visualizations (montages, average images, dataset statistics) so that I can verify the study's findings.  
- **As a business user,** I want to upload images of cherry leaves so that I can check if they are healthy or infected quickly.  
- **As a user,** I want to see a confidence percentage for the model’s prediction so that I understand how reliable the result is.  
- **As a data analyst,** I want to download a report of my uploaded images and predictions so that I can review the results offline.  

---

## 🚀 Epic 5: Model Evaluation and Deployment  
*As a machine learning engineer, I need to test, deploy, and monitor the model so that it remains reliable in real-world conditions.*  

### **User Stories:**
- **As a machine learning engineer,** I want to evaluate the model's performance on an unseen test set to ensure it meets or exceeds the 97% accuracy requirement.  
- **As a developer,** I want to deploy the trained model into a cloud-based application using Streamlit so that users can interact with it easily.  
- **As a business user,** I want a simple interface where I can upload images and receive real-time predictions.  
- **As a model developer,** I want to log prediction data so that I can track model performance over time and detect potential drift.  
- **As a researcher,** I want to test the system with new unseen images (external dataset) to confirm its generalization.  

---

## 🔒 Epic 6: Ethical Considerations and Compliance  
*As a project lead, I need to ensure compliance with data protection policies so that client data remains secure.*  

### **User Stories:**
- **As a data engineer,** I want to store and process all client-provided data under an NDA to protect confidentiality.  
- **As a machine learning developer,** I want to avoid dataset biases that could impact predictions, ensuring fair and transparent results.  
- **As a project manager,** I want to provide a hypothesis explaining the project so that stakeholders understand its scope.  

---

## 📈 **Key Results**
- The **color model** achieved a **test set accuracy exceeding 97%**, meeting the client's expectations.
- The **grayscale model** marginally improved accuracy and loss and file size and would be worth exploring further as a scaleable solution for larger datasets.
- The charts show an accurate result and there are no signs of over or underfitting, this was achieved using stop loss within the training models.

---

## 🔧 **Unfixed Bugs**
1. **Prediction Confidence Display Issue**:  
   When uploading a healthy leaf for analysis, the result intermittently suggests a 0% confidence of accuracy. This does not impact the accuracy which is correct but requires    
   refinement in future iterations. I was unable to find a soloution to this bug.
   
2. **Scalability for Multi-Class Problems**:  
   While the system performs well for binary classification, future scalability to multi-class problems will require further testing and optimization and the use for example, of 
   softmax. I would also evaluate using a confusion matrix for future iterations where datasets are less balanced or contain harder to classify examples.
 
---

## 🚀 **Deployment Instructions**
The project was deployed using **Render** for a seamless and scalable web service.

### Deployment Steps:
1. **Create a Render account** and connect it to the GitHub repository.
2. **Specify the build command**:
   pip install -r requirements.txt
   streamlit run app.py
3. Deploy and test the live app link.

## 🧰 **Technologies Used**
- **Programming Language:** Python
- **Libraries and Frameworks:**
  - **TensorFlow/Keras:** Model training and evaluation (eg. used to train models in the notebooks for Model evaluating and feature reduction).
  - **Pandas and NumPy:** Data preprocessing and manipulation (Eg. Data tables on the ML performance page).
  - **Matplotlib, Seaborn, Plotly:** Data visualizations (used for charts on the ML performance page).
  - **Streamlit:** Interactive dashboard development.
  - **Deployment Platform:** Render for deployment

---

# Testing
- The project passes the Code Institute Python Linter with no errors and is PEP 8 compliant.

---

## 📜 **License**
This project was conducted under an NDA and is proprietary to Farmy & Foods. Unauthorized sharing of the dataset or codebase is prohibited.
The restricted data is saved in a .git ignore file.

---

## 🤝 **Credits**
### Content:
- **Dataset:** [Kaggle Cherry Leaves Dataset](https://www.kaggle.com/codeinstitute/cherry-leaves)
- **Tutorials and Resources:**
- TensorFlow [tutorials](https://www.tensorflow.org/guide/keras/sequential_model).
- Code Institute Malaria Detector walkthrough.
- Coursera - [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction).
- Emoji use - [Streamlit] (https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/).
---

## 🗺️ **Future Plans**
1. **Generalization Testing:**  
   Test the model with external datasets to ensure robustness across different farms and conditions.

2. **Scalability for Multi-Class Problems:**  
   Future versions could incorporate **softmax activation** to classify multiple plant diseases, expanding beyond binary classification and allowing for different species to be 
   tested such as peach trees.

3. **IoT and Mobile Solutions:**  
   Deploy lightweight versions of the model for in-field detection using mobile devices.

4. **Expand to Other Crops:**  
   Extend the solution to detect pests and diseases in other crops, leveraging the success of this project.
