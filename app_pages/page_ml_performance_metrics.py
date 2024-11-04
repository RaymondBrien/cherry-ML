import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from matplotlib.image import imread
from src.machine_learning.eval import load_test_evaluation


def page_ml_performance_metrics():
    """
    Display ML performance page body
    Detail Train Val Test split ratios
    """
    version = 'v6' # TODO edit version if required after final training
    class_distribution = plt.imread(
        f"outputs/{version}/class_distribution.png")

    st.image(class_distribution,
             caption='Labels Distribution on Train, Validation and Test Sets')


    st.header("ML Performance Metrics")
    st.write("### Train, Validation and Test Set: Labels Frequencies")

    st.write(
        "The dataset was split into 3 sets: Train, Validation and Test according "
        "to the following ratios:\n"
        # TODO fill in missing percentages
        "* Set: % \n"
        "* Set: % \n"
        "* Set: % \n"
    )
    st.write("Dataset breakdown")
    # TODO include images representing raw balance of classes
    st.info(
        # TODO make this a DF and write total ratios
        "* Train - healthy: 1472 images\n"
        "* Train - powdery_mildew: 1472 images\n"
        "* Validation - healthy: 210 images\n"
        "* Validation - powdery_mildew: 210 images\n"
        "* Test - healthy: 422 images\n"
        "* Test - powdery_mildew: 422 images\n"
        "* **4208 images total**"
    )
    st.write("---")

    st.write("### Model History")
    st.info(
        "Detailed by the graph below, the learning cycle for this "
        "binary classification model demonstrates: "
        "* a combination of good accuracy and low loss after epoch four during training.\n"
        # TODO how many for train?
        "* As expected, given the very small dataset of X images in the train set total "
        "the small, simple model with 3 lightweight convolutional layers generalises well to the given data"
        "From the documented training accuracy and loss plots, a normal learning curve is demonstrated. \n"
        "The model is neither overfitting or underfitting.\n"

        # TODO Show model outline png
        "Early stopping was also invoked for streamlined learning, defined by accuracy metrics.\n"


    )

    col1, col2 = st.columns(2)
    with col1:
        model_acc = plt.imread(f"outputs/{version}/model_training_acc.png")
        st.image(model_acc, caption='Model Training Accuracy')
    with col2:
        model_loss = plt.imread(f"outputs/{version}/model_training_losses.png")
        st.image(model_loss, caption='Model Training Losses')
    st.write("---")

    st.write("### Generalised Performance on Test Set")
    st.dataframe(pd.DataFrame(
        load_test_evaluation(version),
        index=['Loss', 'Accuracy']))

    st.success(
        # TODO update
        f"**The general accuracy of ML model is 99.76%!!** "
    )

    st.write("---")
    st.write("Further info:")
    st.info(
        # TODO show train and validation evaluation results too, generated from their pkl files
        f"The ML model was trained on a dataset consisting of 4208 cherry leaves. The model was evaluated on a separate set of 422 cherry leaves.\n"
        f"The model achieved an accuracy of 99.76% on the test set.\n"
        f"This indicates that the model is capable of predicting the presence of powdery mildew in cherry leaves with a high degree of accuracy.\n"
    )
    load_test_evaluation(version)