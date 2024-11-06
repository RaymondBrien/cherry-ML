import random
import pickle
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

from matplotlib.image import imread
from tensorflow.keras.preprocessing import image
from src.machine_learning.eval import load_evaluation

def page_ml_performance_metrics():
    """
    Display ML performance page body
    Detail Train Val Test split ratios
    """
    version = 'v6'
    class_distribution = plt.imread(
        f"outputs/{version}/class_distribution.png")

    st.header("ML Performance Metrics")
    st.write("### Train, Validation and Test Set: Labels Frequencies")

    st.image(class_distribution,
             caption='Labels Distribution on Train, Validation and Test Sets')
    st.write(
        "The dataset was split into 3 sets: Train, Validation and Test according "
        "to the following ratios:\n"
        
        "* Train Set: 70% \n"
        "* Validation Set: 10% \n"
        "* Test Set: 20% \n"
    )
    st.write("Dataset breakdown")

    data = {
        "Set": ["Train", "Train", "Validation", "Validation", "Test", "Test", "Total"],
        "Label": ["Healthy", "Powdery Mildew", "Healthy", "Powdery Mildew", "Healthy", "Powdery Mildew", "All Images"],
        "Count": [1472, 1472, 210, 210, 422, 422, 4208]
    }

    df = pd.DataFrame(data)

    # Filter out the "Total" row for individual counts by set and label
    df_plot = df[df["Set"] != "Total"]

    # Create the Plotly bar chart
    fig = px.bar(
        df_plot, 
        x="Set", 
        y="Count", 
        color="Label", 
        barmode="group",
        title="Image Count by Set and Label",
        labels={"Count": "Image Count", "Set": "Dataset Partition"},
        text="Count"
    )

    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(yaxis=dict(title="Image Count"))

    st.plotly_chart(fig)

    st.write(f"**Total Images:** {df.loc[df['Set'] == 'Total', 'Count'].values[0]}")


    st.write("---")

    st.write("### Model History")
    st.info(
        "Detailed by the graph below, the learning cycle for this "
        "binary classification model demonstrates: "
        "* a combination of good accuracy and low loss after epoch four during training.\n"
        "* As expected, given the very small dataset of 2944 images in the train set total "
        "the small, simple model with 3 lightweight convolutional layers generalises well to the given data"
        "From the documented training accuracy and loss plots, a normal learning curve is demonstrated. \n"
        "The model is neither overfitting or underfitting.\n"

        "Early stopping was also invoked for streamlined learning, defined by accuracy metrics.\n"

        "- The image dataset is overall very small for a CNN. Whilst developing the model,"
        "overfitting was likely and Image augmentation was neccessary for effective"
        "learning. It is believed this is the reduced complexity model worked better.\n"

        "Hyperparamter optimisation was used during earlier versions of model training "
        "using the keras hyberband tuner "
        "[Keras Tuner](https://www.tensorflow.org/tutorials/keras/keras_tuner) "
        "however, the gitpod cloud IDE made such training impossible givne the unalterable 15 minute timeout."
        "Furthermore, 99.88 percent accuracy was achieved for the final model iteration and "
        "therefore proves that hypertuning for this particular ML problem would not have added "
        "much value to the task. "


    )
    


    # show model acc 
    model_acc = plt.imread(f'outputs/{version}/model_training_acc.png')
    st.image(model_acc, caption='Model Training Accuracy')
    
    # show model loss
    model_loss = plt.imread(f'outputs/{version}/model_training_losses.png')
    st.image(model_loss, caption='Model Training Losses')
    
    st.subheader('Model Outline:')
    st.image(f'outputs/{version}/model_summary.png')
    st.write("---")

    st.write("### Generalised Performance on Test Set")
    data = pd.DataFrame(
        load_evaluation(version, dataset='test'),
        index=['Loss', 'Accuracy'])
    
    # show stats df
    stats = st.dataframe(data)
    # to dynamically update text for future models
    accuracy_reading = format((data.at['Accuracy', 'test']*100), '.2f')

    st.success(
        f"**The general accuracy of ML model is {accuracy_reading}%** "
    )

    st.write("---")

    st.write("Further info:")
    st.info(
        "The ML model was trained on a dataset consisting of 4208 cherry leaves in total. "
        "Testing involved an unseen set of 422 cherry leaves, split during the data preparation phase.\n"
        f"The model achieved an accuracy of {accuracy_reading}% on the test set.\n"
        "This indicates that the model is capable of predicting the presence of powdery mildew in cherry "
        "leaves with a high degree of accuracy, and able to achieve the required performance metrics.\n"
    )

    st.write('Extra Stats: validation evaluation')
    # display validation evaluation as streamlit dataframe
    st.dataframe(pd.DataFrame(
        load_evaluation(version, dataset='val'),
        index=['Loss', 'Accuracy']))
