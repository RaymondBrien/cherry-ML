### Powdery Mildew in Cherry Leaves Detector

Powdery Mildew in Cherry Leaves Detector is a machine learning app designed to predict whether a cherry leaf is healthy or infected with powdery mildew, a common fungal disease. By processing images of cherry leaves, the app performs a binary classification to help determine the health status of a given leaf.

[View the live project here](https://cherry-ml-2c1328018520.herokuapp.com/)

TODO add image

---

## **Table of Contents**
1. [Business Context](#business-context)
2. [Dataset](#dataset)
3. [Business Requirements](#business-requirements)
4. [Project Hypothesis](#project-hypothesis)
5. [Model Design and Metrics](#model-design-and-metrics)
6. [Dashboard Design](#dashboard-design)
7. [Technologies Used](#technologies-used)
8. [Deployment](#deployment)
9. [Credits and Acknowledgments](#credits-and-acknowledgments)

---

## **1. Business Context**

Powdery mildew is a parasitic fungal disease caused by *Podosphaera clandestina* in cherry trees. Infected plants display powdery white spots on leaves and stems, which can significantly impact the quality and yield of cherry crops. Currently, Farmy & Foods, a cherry plantation company, conducts manual inspections to detect powdery mildew—a process that takes approximately 30 minutes per tree. This manual method is time-consuming and unscalable, leading to inefficiencies in large-scale operations.

This ML-based app was developed to streamline the detection process. By uploading images of cherry leaves, staff can obtain an instant health assessment of each leaf, allowing for quicker and more effective disease management.

---

## **2. Dataset**

- **Source**: The dataset was retrieved from [Kaggle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves) and contains 4,208 images (256x256 pixels each) of both healthy and powdery mildew-infected cherry leaves.
- **Image Composition**: The dataset is equally weighted with images of healthy and infected leaves (50% each).
- **Purpose**: This dataset allows us to build a binary classification model to automate the identification of powdery mildew infection.

---

## **3. Business Requirements**

The client, Farmy & Foods, has two principal requirements:
  1 - Visually differentiate between healthy and infected cherry leaves.
  2 - Use image-based predictions to identify powdery mildew infection in cherry leaves quickly and accurately.

By implementing this ML workflow, Farmy & Foods hopes to reduce inspection time and improve crop quality through early detection and treatment of infections.

---

## **4. Project Hypothesis**

- **Hypothesis**: Cherry leaves affected by powdery mildew display unique visual characteristics such as a lighter color and fine powdery patches. These features are distinguishable through image processing techniques and can be leveraged to train an ML model.
- **Validation**:
   - **Image Montage**: To visually illustrate the differences between healthy and infected leaves.
   - **Average and Variability Images**: Calculating the average image for both healthy and infected leaves to identify any distinct color or texture patterns.
   - **Difference Between Averages**: Comparing the average images of healthy and infected leaves to highlight visual differences.

The ML model should achieve a target accuracy of at least 97% in correctly classifying leaf images as healthy or infected.

---

## **5. Model Design and Metrics**

This project utilizes a Convolutional Neural Network (CNN) model built with TensorFlow and Keras. Given the visual complexity of distinguishing powdery mildew from healthy leaves, a CNN is well-suited for this image classification task.

- **Model Structure**: The model has approximately 6 million parameters, trained on a dataset of 4,208 labeled images. It uses a binary classification approach to predict the likelihood of powdery mildew presence.
- **Performance Metrics**:
   - **Training Accuracy**: The model achieved high accuracy on the training set, consistently above 97%.
   - **Generalized Accuracy**: On the test set, the model maintained an accuracy of 99.76%, exceeding the required threshold.
   - **Loss & Accuracy Over Epochs**: Detailed loss and accuracy plots showcase the model’s performance during training.

---

## **6. Dashboard Design**

The app dashboard provides an intuitive interface with five main pages for streamlined use and understanding of the model’s predictions and performance metrics:

### **Page 1: Quick Project Summary**
   - **Overview**: Brief introduction to powdery mildew and its impact on cherry plantations.
   - **Dataset Summary**: Information about dataset size, source, and class balance.
   - **Requirements Overview**: Displays the project’s business goals and ML objectives.

### **Page 2: Leaves Visualizer**
   - **Visual Analysis Options**:
       - Mean and standard deviation displays for healthy and infected leaves.
       - Difference between average healthy and infected leaf images.
       - Image montage of healthy and powdery mildew-infected leaves for side-by-side comparison.

### **Page 3: Powdery Mildew Detector**
   - **File Upload and Prediction**: Users can upload images for instant analysis.
   - **Prediction Output**: Displays a statement indicating whether the uploaded image shows a healthy or infected leaf.

### **Page 4: Project Hypothesis and Validation**
   - **Hypothesis Display**: Explanation of visual distinctions between healthy and infected leaves.
   - **Validation Methods**: Visual comparisons and metrics proving the model’s accuracy.

### **Page 5: ML Performance Metrics**
   - **Label Frequency**: Shows the distribution of healthy and infected samples across training, validation, and test sets.
   - **Model History**: Plots for accuracy and loss over epochs.
   - **Evaluation Results**: Summary of model performance with metrics such as accuracy, loss, and F1 score.

---

## **7. Technologies Used**

### Languages
- **Python**

### Frameworks and Libraries
   - **TensorFlow/Keras**: For building and training the CNN model.
   - **NumPy**: Array operations.
   - **Pandas**: Data manipulation.
   - **Matplotlib & Seaborn**: Data visualization.
   - **Streamlit**: Web application development for interactive deployment.
   - **Plotly**: Interactive plots in Jupyter notebooks.
   - **PIL**: Image processing.
   - **Joblib**: Saving/loading model and pipeline.

### Deployment Tools
   - **Git/GitHub**: Version control.
   - **Heroku**: Deployment platform.
   - **Balsamiq**: Used to create wireframes.

---

## **8. Deployment**

The app is hosted on Heroku, following these deployment steps:
   - **Connect GitHub Repository**: Link the Heroku app to the project’s GitHub repository.
   - **Set Deployment Branch**: Select the branch for deployment.
   - **Deploy Branch**: Deploy directly from GitHub.
   - **Configure Stack**: Adjust Heroku stack version as required (resolved Python compatibility issues by switching stack).
   - **Slug Size**: Managed large files by adding them to `.slugignore`.

App live link: [https://cherry-leaf-mildew-detector.herokuapp.com/](https://cherry-leaf-mildew-detector.herokuapp.com/)

---

## **9. Credits and Acknowledgments**

- **Project References**: 
   - [Code Institute Malaria Walkthrough Project](https://learn.codeinstitute.net/courses/course-v1:code_institute+CI_DA_ML+2021_Q4/courseware/07a3964f7a72407ea3e073542a2955bd/29ae4b4c67ed45a8a97bb9f4dcfa714b/)
   - Other project repositories: [GyanShashwat1611/WalkthroughProject01](https://github.com/GyanShashwat1611/WalkthroughProject01), [HaimanotA/Instant-Mildew-Detector](https://github.com/HaimanotA/Instant-Mildew-Detector), and [alerebal/Powdery Mildew](https://github.com/Code-Institute-Submissions/milestone-project-mildew-detection-in-cherry-leaves.git)

- **Content Sources**:
   - Streamlit documentation
   - Wikipedia (for understanding powdery mildew infection)
   - Code Institute Slack community

- **Mentorship**: Special thanks to my mentor, **Precious Ijege**, for their invaluable support and guidance throughout the project.

--- 

This README covers both the technical and practical details, creating a thorough resource for understanding and utilizing the Powdery Mildew in Cherry Leaves Detector app.