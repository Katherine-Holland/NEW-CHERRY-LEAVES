READ ME

## Project Overview
This project demonstrates how predictive analytics can optimize agricultural operations. The focus is on addressing a fictitious scenario where Farmy Foods faces powdery mildew issues in their cherry plantations, a key product in their portfolio. The project aims to minimize revenue loss from poor-quality produce and streamline the monitoring process, which is currently manual and unsustainable.

The cherry plantation crop from Farmy & Foods is facing a challenge where their cherry plantations have been presenting powdery mildew. Currently, the process is manual verification if a given cherry tree contains powdery mildew. An employee spends around 30 minutes in each tree, taking a few samples of tree leaves and verifying visually if the leaf tree is healthy or has powdery mildew. If there is powdery mildew, the employee applies a specific compound to kill the fungus. The time spent applying this compound is 1 minute. The company has thousands of cherry trees located on multiple farms across the country. As a result, this manual process is not scalable due to the time spent in the manual process inspection.

To save time in this process, the IT team suggested an ML system that detects instantly, using a leaf tree image, if it is healthy or has powdery mildew. A similar manual process is in place for other crops for detecting pests, and if this initiative is successful, there is a realistic chance to replicate this project for all other crops. The dataset is a collection of cherry leaf images provided by Farmy & Foods, taken from their crops.

## Business Case Understanding and Outcomes Required
This project follows the CRISP-DM Workflow (Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, Deployment). 

We have decided that this project would benefit from a ML solution to inform the client whether or not cherry leaves are infected with powdery mildew.
Data has already been collected and labelled and the data set is balanced for both classes, heathy and infected leaves.
The client has requested a dashboard rather than relying on API endpoints, with the required business outputs:
1) Differentiating visually between healthy and infected cherry leaves:
An image montage for either healthy or leaves infected with powdery mildew.
Visualization of the average image and standard deviation image for healthy and infected leaves
Visualization of the difference between an average leaf infected with powery mildew and an average healthy leaf.
2) Ability to predict quickly with a >97% accuracy if a cherry leaf is healthy or infected, based on an uploaded image.

This initiative is scalable and could extend to other crops, enhancing operational efficiency and market quality.
This project dataset is to be kept private.

## The rationale to map the business requirements to the Data Visualisations and ML tasks

We have identified 2 business requirements and the ML Business case:

### Business Requirement 1 
### Visual Differentiation
- Generate visualizations comparing average healthy and infected leaves.
- Create montages to highlight class differences.


### Business Requirement 2 
### Classification
- Develop a neural network model capable of distinguishing between healthy and infected leaves.
- Target accuracy: ≥97% for effective real-world deployment.
- Performance validation through k-fold cross-validation and confusion matrix analysis, identifying false positives/negatives to fine-tune the model.

## ML Business Case

* We want to use a deep learning - convolutional neural network ML model to predict if a leaf is infected with powdery mildew or not, based on historical image data. It is a supervised model, a 2-class, single-label, classification model. The data would align with a binary classifier, indicating whether a particular cherry leaf is healthy or contains powdery mildew.
- The model input is a cherry leaf image and the model output is a prediction of whether the cherry leaf is healthy or contains powdery mildew.

Heuristics: 
The current way to diagnose requires experienced staff and lengthly inspection times across orchards to distinguish healthy or infected leaves. This process is not scalable.

### Epics and User Stories
Following the Agile development principles we can create a workflow by dividing up the project. 

Stream lit user stories:

As a first time visitor i want to easily understand the organisations aims for the project

As a returning visitor i want to be able to upload my images to the detector to be analysed and get a prediction

STREAMLIT filters- image montage, average image and variability, differences in healthy and infected leaves (radio buttons or drop down.


## Dataset Content

The dataset, sourced from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves) contains 4,208 images of cherry leaves categorized as healthy or infected with powdery mildew, a fungal disease affecting many plants. 
We have extracted all images from this dataset and added to this with synthetic data.

Does the data suggest a particular model such as regression, classification or clustering? 

This project looks at cherry leaves and whether they are healthy or not. The data is alreafy labelled and therefore  are categorial. As this is a categorical variable then this is a classification problem. The metrics will be accuracy, recall, precision, F1 or specificity. As the data set class samples are equal and the cost of misclassification is low, this would be a good case for a model focussed on accuracy. Also as recal and precision are valued we could also use the F1 metric.

Heuristics
It would be worthwhile to test if the model could be trained to test greyscale 
Soft max used over sigmoid for the reasons.. …
confusion matrix

The data is already labelled healthy or not and we do not need to do any furhter feature engineering as we are using a deep neural network.

## Hypothesis and how to validate

## Data visualization, cleaning, and preparation.
Hypothesis: Powdery mildew manifests visibly through signs like shriveling, white discoloration, or spotting on leaves.

Validation Steps:
- Use the dataset to train a binary classification model.
- Conduct image analysis to confirm visual differences.
- Evaluate the model using metrics like accuracy, precision, recall, and F1 score, aiming for >97% accuracy.
- If classes (helathy vs. infected) are imbalanced we will apply StratifiedK-fold cross-validation to ensure each fold has a proportional representation of both classes.

Dataset Preparation:

Compile a labeled dataset of cherry tree leaves, categorized into "Healthy" and "Infected" classes.
Ensure the dataset is diverse, representing various lighting conditions, angles, and leaf conditions. If the dataset is too small create extra images.

## Model training, optimization and validation

Model Training and Testing:
Train a neural network model on the dataset using a supervised learning approach.
Split the dataset into training, validation, and test sets (e.g., 70%, 20%, 10%).

Evaluation Metrics:
Use precision, recall, F1 score, and accuracy to measure the model's performance on the test set.
Data Augmentation: To ensure robust data augmentation techniques will be applied during training. Examples include random rotations, flips, brightness adjustments, and zooms to simulate real-world variations and increase model generalizability. Apply a confusion matrix to analyze false positives and false negatives.

Visual Inspection:
Validate predictions by visually inspecting misclassified images to identify patterns or potential dataset biases.

Cross-Validation:
Perform k-fold cross-validation to ensure robustness and generalizability of the model.

Hypothesis Support:
If the model achieves high accuracy (e.g., >97%) in distinguishing healthy and infected leaves, the hypothesis is supported.
If accuracy is low, investigate feature extraction, dataset quality, or other confounding variables.



Dashboard planning, designing, and development.
## Dashboard Design (Streamlit App User Interface)

### Page 1: Project Summary
* Quick project summary
    * General Information
        * What is Powdery mildew and how does it present itself in cherry tree leaves.
        * Usually identification of infected leaves is a manual process conducted by the employee of the orchard.
        * Stats on numbers of trees infected.
    * Project Dataset
        * The available dataset contains 4k images of leaves.
    * Link to addition information
    * Business requirements
        *  The client is interested in ...
        *  The client is interested to know...

### Page 2: Leaf Visualiser
* It will answer business requirement 1
    * Checkbox 1 - Difference between average and variability image
    * Checkbox 2 - Differences between average mildew and average healthy leaves
    * Checkbox 3 - Image Montage

### Page 3: Mildew Detector
* Business requirement 2 information - "The client is interested to tell whether a leaf is infected with powdery mildew or not.
* Link to download a set of mildew infected and healthy leaf images for live prediction.
* User Interface with a file uploader widget. The user should upload multiple leaf images. It will display the image and a prediction statement, indicating if the leaf is infected or not with mildew and the probability associated with this statement. 
* Table with image name and prediction results.
* Download button to download table.

### Page 4: Project Hypothesis and Validation
* Bloack for each project hypothesis, describe the conclusion and how you validated.

### Page 5: ML Performance Metrics
* Label Frequencies for Train, Validation and Test Sets
* Model History - Accuracy and Losses
* Model evaluation result

## Unfixed Bugs

- You will need to mention unfixed bugs and why they were unfixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable for consideration, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed.

## Set Up Process and Deployment
Dashboard deployment and release.
### Set up the project
Create a repository and download dependencies.

## Download the dataset from Kaggle.
Ensure you have an authentication key (JSON file) generated from your Kaggle account. Using the kaggle API endpoint you will use this to download and extract the dataset.
Drag and drop the JSON file into the workspace. See the DataCollection Jupyter notebook for further instructions.

### Render

- The App live link is: `https://YOUR_APP_NAME.herokuapp.com/`
- Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
- The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large, then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries

- Here, you should list the libraries used in the project and provide an example(s) of how you used these libraries.
- It should be noted that the client provided the data under an NDA (non-disclosure agreement), therefore the data should only be shared with professionals that are officially involved in the project. To comply with this the data has been added to the git.ignore file.

## Credits

TensorBoard
https://github.com/tensorflow/tensorboard/blob/master/README.md
Tutorial:  https://www.tensorflow.org/tensorboard/get_started

Code Institute walkthough project 01

### Content

- The text for the Home page was taken from Wikipedia Article A.
- Instructions on how to implement form validation on the Sign-Up page were taken from [Specific YouTube Tutorial](https://www.youtube.com/).
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/).

### Media

- The photos used on the home and sign-up page are from This Open-Source site.
- The images used for the gallery page were taken from this other open-source site.

## Acknowledgements (optional)

- Thank the people who provided support throughout this project.

## Future Plans
1) Integration of Weather Data:
Utilize platforms like AgWeatherNet to correlate weather conditions with powdery mildew outbreaks, enabling predictive interventions, as these outbreaks tend to occur on new season buds and leaves (following rainfall) and on the mature fruit prior to harvest.

2) Mobile and IoT Solutions:
Deploy lightweight versions of the model on mobile apps or IoT devices for in-field detection, reducing latency and reliance on centralized processing.

3) Dataset Expansion:
Extend the dataset to include other crops like apples and peaches, enabling broader applications across Farmy Foods' operations.