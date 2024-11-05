### Powdery Mildew in Cherry Leaves Detector

Powdery Mildew in Cherry Leaves Detector is a machine learning app designed to predict whether a cherry leaf is healthy or infected with powdery mildew, a common fungal disease. By processing images of cherry leaves, the app performs a binary classification to help determine the health status of a given leaf.

[View the live project here](https://cherry-ml-2c1328018520.herokuapp.com/)

<!-- TODO add image -->

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

## **Business Context**

Powdery mildew is a parasitic fungal disease caused by *Podosphaera clandestina* in cherry trees. Infected plants display powdery white spots on leaves and stems, which can significantly impact the quality and yield of cherry crops. Currently, Farmy & Foods, a cherry plantation company, conducts manual inspections to detect powdery mildew—a process that takes approximately 30 minutes per tree. This manual method is time-consuming and unscalable, leading to inefficiencies in large-scale operations.

---

## **Dataset**

- **Source**: The dataset was retrieved from [Kaggle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves) and contains 4,208 images (256x256 pixels each) of both healthy and powdery mildew-infected cherry leaves.
- **Image Composition**: The dataset is equally weighted with images of healthy and infected leaves (50% each).
- **Purpose**: This dataset allows us to build a binary classification model to automate the identification of powdery mildew infection.

---

## **Business Requirements**

The client, Farmy & Foods, has two principal requirements:
  1 - Visually differentiate between healthy and infected cherry leaves.
  2 - Use image-based predictions to identify powdery mildew infection in cherry leaves quickly and accurately.

By implementing this ML workflow, Farmy & Foods hopes to significantly reduce inspection time and improve crop quality through early detection and treatment of infections.

### * Rationale to map the business requirements to the Data Visualizations and ML tasks

- Business requirement 1:
  - This is not an ML model problem but a data exploration and visualisation problem. Through providing
   exploratory image montages and dataset analysis we prepare the dataset for efficient model training.

- Business requirement 2:
  - This ML-based app was developed to streamline the detection process. By uploading images of cherry leaves, staff can obtain an instant health assessment of each leaf, allowing for quicker and more effective disease management.
  - This is a binary classification problem. Given its available datasources, a CNN will be suitably equipped for developing a suitable model.
  - We want the model to predict the likelihood as a percentage that a given leaf is infected or not, based on the provided image dataset from the client. The ideal outcome is for the model to score at least 97% in accuracy, providing a clear and well generalised understanding of the image data patterns between the two classes (healthy and infected), providing not only the answer to which class an unseen leaf image likely lies, but also how likely it is to belong to that class. The model output is a class prediction and a likelihood of belonging to that class as a float.
  - The training data is split into precisely equal classes (healthy, infected), totalling 70% of the total dataset image count. Image labels are assigned from the given file structure, due to the raw dataset organisation already having structured the total dataset into the two classes.

---

## **Project Hypothesis**

- **Hypothesis 1**: It is hypothesized that infected cherry tree leaves will be visually differentiable from healthy cherry tree leaves. Specifically, white powdery mildew will be present on the surface of
the majority of infected leaves within the dataset.
- **Validation**:
  - **Image Montage**: To visually illustrate the differences between healthy and infected leaves.
  - **Average Image Per Class**: Calculating the average image for both healthy and infected leaves to identify any distinct color or texture patterns.
  - **Difference Between Averages**: Comparing the average images of healthy and infected leaves to highlight visual differences.

- **Hypotheesis 2**: It is hypothesised that using only the provided dataset, the ML model will be able to distinguish between a healthy cherry leaf and an infected cherry leaf with at least 97% accuracy.
- **Validation**:
   - **Accuracy Evaluation on Test Set**: After model training, model evaluation will measure accuracy and loss readings. Accuracy above 97% per business requirement 2, will pass.

<!-- TODO add hypo 3 -->

---

## **Model Design and Metrics**

This project utilizes a Convolutional Neural Network (CNN) model built with TensorFlow and Keras. Given the visual complexity of distinguishing powdery mildew from healthy leaves, a CNN is well-suited for this image classification task.

<!-- TODO update params number below -->
- **Model Structure**: The model has approximately x params, trained on a dataset of 4,208 labeled images. It uses a binary classification approach to predict the likelihood of powdery mildew presence.
- **Performance Metrics**:
  - **Training Accuracy**: The model achieved high accuracy on the training set, consistently above 97%.
  <!-- TODO accuracy % -->
  - **Generalized Accuracy**: On the test set, the model maintained an accuracy of 99.76%, exceeding the required threshold.
  - **Loss & Accuracy Over Epochs**: Detailed loss and accuracy plots showcase the model’s performance during training.

<!-- TODO add image of model outline -->
<!-- TODO add specific choices and why and sources for activation etc -->
---

## **Dashboard Design**

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

## **Technologies Used**

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

---

## **Deployment**

The app is hosted on Heroku, following these deployment steps:

- **Connect GitHub Repository**: Link the Heroku app to the project’s GitHub repository.
- **Set Deployment Branch**: Select the branch for deployment.
- **Deploy Branch**: Deploy from linked GitHub repository.
- **Configure Stack**: Adjust Heroku stack version as required (resolved Python compatibility issues by switching to stack 20).
> [!TIP]
> - Via heroku CLI: 
> Config via API key
> `heroku config (use personal API link)`
> set stack to stack 20
> `heroku stack:set heroku-20 -a [app_name] `

- **Slug Size**: Managed large files by adding them to `.slugignore`.

[App live link](https://cherry-ml-2c1328018520.herokuapp.com/)

---

## **Credits and Acknowledgments**

- **Project References**:
  - [Code Institute Malaria Walkthrough Project](https://learn.codeinstitute.net/courses/course-v1:code_institute+CI_DA_ML+2021_Q4/courseware/07a3964f7a72407ea3e073542a2955bd/29ae4b4c67ed45a8a97bb9f4dcfa714b/)
  - Other project repositories: [GyanShashwat1611/WalkthroughProject01](https://github.com/GyanShashwat1611/WalkthroughProject01), [HaimanotA/Instant-Mildew-Detector](https://github.com/HaimanotA/Instant-Mildew-Detector), and [alerebal/Powdery Mildew](https://github.com/Code-Institute-Submissions/milestone-project-mildew-detection-in-cherry-leaves.git)

- **Content Sources**:
  - Streamlit documentation
  - Wikipedia (for understanding powdery mildew infection)
  - Code Institute Slack community

- **Mentorship**: Special thanks to my mentor, **Precious Ijege**, for their invaluable support and guidance throughout the project.

---
