# Cherry-ml PP5
<!-- TODO add badges -->

## Dataset Content

- The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/cherry-leaves). We then created a fictitious user story where predictive analytics can be applied in a real project in the workplace.
- The dataset contains +4 thousand images taken from the client's crop fields. The images show healthy cherry leaves and cherry leaves that have powdery mildew, a fungal disease that affects many plant species. The cherry plantation crop is one of the finest products in their portfolio, and the company is concerned about supplying the market with a compromised quality product.

## Business Requirements

The cherry plantation crop from Farmy & Foods is facing a challenge where their cherry plantations have been presenting powdery mildew. Currently, the process is manual verification if a given cherry tree contains powdery mildew. An employee spends around 30 minutes in each tree, taking a few samples of tree leaves and verifying visually if the leaf tree is healthy or has powdery mildew. If there is powdery mildew, the employee applies a specific compound to kill the fungus. The time spent applying this compound is 1 minute. The company has thousands of cherry trees located on multiple farms across the country. As a result, this manual process is not scalable due to the time spent in the manual process inspection.

To save time in this process, the IT team suggested an ML system that detects instantly, using a leaf tree image, if it is healthy or has powdery mildew. A similar manual process is in place for other crops for detecting pests, and if this initiative is successful, there is a realistic chance to replicate this project for all other crops. The dataset is a collection of cherry leaf images provided by Farmy & Foods, taken from their crops.

1. The client is interested in conducting a study to visually differentiate a healthy cherry leaf from one with powdery mildew.
2. The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.

## Hypothesis and how to validate?

<!-- TODO List here your project hypothesis(es) and how you envision validating it (them). -->

- We suspect leaves containing powdery mildew (from here on referred to as 'infected') will show visual and recognisable signs of the mildew, differentiating them from healthy leaves.
  - An average image study can help to uncover and investigate these differences.

## The rationale to map the business requirements to the Data Visualisations and ML tasks

<!-- - List your business requirements and a rationale to map them to the Data Visualisations and ML tasks. -->

- Business Requirement 1: Data Visualisation

  - Mean-average and standard deviation representations to assess variability of images will be displayed for both infected and uninfected leaf classes (healthy or powdery mildew).
  - The differences between an average infected leaf image and an uninfected leaf image will be displayed and defined.
  - An image montage of both classes (infected and uninfected) will be collated for a clear visual representation of each class.

- Business Requirement 2: Binary Classification using Convolutional Neural Networks

  - We aim to predict if a given leaf is infected or not judging by the presence of powdery mildew.
  - We aim to use the CNN to map relationships between features and labels.
  - We aim to build a binary classifier and generate reports.

## ML Business Case

### MildewClf
<!-- TODO confirm this is correct after initial data inspection -->
<!-- TODO change remove either 50x50 or 100x100 once decided  -->
- We want an ML CNN network to predict if a leaf contains powdery mildew, (indicating infection) based on historical image data. It is a supervised model, a 2-class, single-label classification model. Images will first be transformed to 100x100px (**or 50x50px**) for equal image handling.

Our ideal outcome is to provide the client with a faster and more reliable diagnostic for mildew detection.

- The model success metrics are:
<!-- TODO aim for 98% or more -->
  - Accuracy of 97% or above on the test set.
  - The model output is defined as a flag, indicating if the leaf displays signs of mildew and the model's calculated probability of infection.
  - Input: leaf image.
  - Output: Prediction (per image) of whether the cherry leaf is healthy or contains mildew.

- Heuristics:
The current diagnostic needs an experienced staff and detailed inspection to distinguish infected and non-infected leaves. Trees are inspected manually, with each inspection per tree averaging 30 minutes: this includes taking sample tree leaves and assessing from basic visual criteria to determine if a leaf is infected. If mildew is present, an employee will apply antifungal compounds to kill the mildew, for each tree taking c.1 minute.
This leaves room to produce inaccurate diagnostics due to human error. Furthermore, staff expertise, training and infection inspections are hugely costly for business, preventing scalable growth due to personel demands and knowledge.
The development of an effective ML model will provide scalable growth opportunities for wider varieties of crops, however, most importantly can give the client the confident ability to supply the market with products which are not of comprimised quality.

- Ethical or Privacy concerns?
The client provided the data under an NDA (non-disclosure agreement), therefore the data should only be shared with professionals that are officially involved in the project.

<!-- TODO determine total image count of dataset UPDATE after initial analysis -->
The dataset is a collection of cherry leaf images provided by Farmy & Foods, taken from their crops. This dataset contains about **1234** thousand images. We have extracted a subset of **1234** images from this dataset and saved it to Kaggle dataset directory for quicker model training.

Train data - target: infected or not; features: all images

## Dashboard Design (Streamlit App User Interface)

<!-- TODO List all dashboard pages and their content, either blocks of information or widgets, like buttons, checkboxes, images, or any other items, that your dashboard library supports. -->
<!-- TODO Finally, during the project development, you may revisit your dashboard plan to update a given feature (for example, at the beginning of the project, you were confident you would use a given plot to display an insight, but later, you chose another plot type). -->

### Page 1: Quick Project Summary

- Brief Project Summary:

<!-- TODO update these questions with content -->
  - General Information:

    - What is mildew? **ANSWER**
    <!-- - TODO LINK TO FURTHER INFO -->
    - Link here

  - Project Dataset:
        <!-- TODO update numbers accordingly -->
    - A leaf image is collected and visually inspected. There may be multiple leaves per tree used for visual inspection, with each leaf considered individually. The dataset is a collection of cherry leaf images provided by Farmy & Foods, taken from their crops. This dataset contains about **1234** thousand images. We have extracted a subset of **1234** images from this dataset and saved it to Kaggle dataset directory for quicker model training.

  - Business Requirements:
    1. The client is interested in conducting a study to visually differentiate a cherry leaf that is healthy from one that contains powdery mildew.
    2. The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew, using a model that reports with a 97% accuracy reading.


### Page 2: Findings Report

- A report detailing the study to visually differentiate a cherry leaf that is healthy from one that contains powdery mildew

### Page 3: Leaves Visualiser

*This will answer business requirement no.1*

- Checkbox 1 - Mean average and variability image differences visualisation
- Checkbox 2 - Differences between infected and uninfected leaves
- Checkbox 3 - Image montage
<!-- TODO add (you may use the Kaggle repository that was provided to you). -->
- A link to download a set of cherry leaf images for live prediction

### Page 4: Infection Detector

*This will answer business requirement no.2*

- Business requirement 2 information: 'The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew'

- A User Interface with a file uploader widget:
  - The user will have the capacity to upload multiple images
  - For each image, it will display the image and a prediction statement, indicating if a cherry leaf is healthy or contains powdery mildew and the probability associated with this statement
  - The User Interface should be intuitive and usable by all employees with only basic technological competency required.
  - A table with the image name and prediction results
    - A download button to download the table

### Page 4: Hypothesis and Validation

<!-- TODO ensure all relevant hypothesis are included!  -->
- Details of each project hypothesis
  - Details of how this was validated across the project

### Page 5: ML Performance Metrics

A technical page displaying model performance:

1. Label Frequencies for Train, Validation and Test Sets
2. Model History - Accuracy and Losses
3. Model evaluation result

## Unfixed Bugs
<!-- TODO document any unfixed bugs -->
<!-- - You will need to mention unfixed bugs and why they were unfixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a significant variable for consideration, paucity of time and difficulty understanding implementation is not a valid reason to leave bugs unfixed. -->

## Deployment

### Heroku
<!-- TODO edit once ready to deploy -->
- The App live link is: `https://cherry-ml.herokuapp.com/`
- Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
- The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large, then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries

<!-- TODO Here, you should list the libraries used in the project and provide an example(s) of how you used these libraries. -->

## Credits

<!-- TODO In this section, you need to reference where you got your content, media and from where you got extra help. It is common practice to use code from other repositories and tutorials. However, it is necessary to be very specific about these sources to avoid plagiarism. -->
<!-- TODO You can break the credits section up into Content and Media, depending on what you have included in your project. -->

### Content Credits
<!-- TODO edit Content Credits -->
<!-- - The text for the Home page was taken from Wikipedia Article A.
- Instructions on how to implement form validation on the Sign-Up page were taken from [Specific YouTube Tutorial](https://www.youtube.com/).
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/). -->

### Media Credits
<!-- TODO edit media credits -->
<!-- - The photos used on the home and sign-up page are from This Open-Source site.
- The images used for the gallery page were taken from this other open-source site. -->

## Acknowledgements (optional)
<!-- TODO add Acknowledgements  -->
- Thank the people who provided support throughout this project.
