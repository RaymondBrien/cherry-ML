import os
import pandas as pd
import streamlit as st

def page_summary_body():

    st.title('Project Summary:')
    st.write('---')

    st.info("**What is Mildew?**\n")
    st.write(
        "Mildew refers to a white powdery fungal growth that can affect "
        "the leaves of plants, including crops. It is a common plant disease "
        "that can reduce the health and productivity of affected plants.\n"
    )
    st.info(
        "**Project Terms and Jargon:**\n"
    )
    st.write(
        "- **Healthy** leaves are those without any visible signs of powdery mildew.\n"
        "- **Unhealthy** leaves are those that show white powdery mildew on its surface, "
        "characteristic of mildew infection.\n"
        "The goal of this project is to build a system that can accurately detect the "
        "presence of powdery mildew on leaf images, in order to help identify and manage "
        "the client's business effectively."
    )
    st.write('---')

    st.subheader('Dataset Summary:')
    df = pd.DataFrame({
        "Total Contents:": ['4208'],
        "Image Size": ['256 x 256 px'],
        "Labels": ['Healthy, Powdery Mildew'],
    })
    st.dataframe(df)
    st.write(
        "Click here for data source: \n"
        "[Kaggle](https://www.kaggle.com/datasets/codeinstitute/cherry-leaves)"
    )

    st.write('---')

    st.subheader("Business Requirements:\n")
    st.subheader("1.\n")

    st.warning(
        "**The client is interested in conducting a study to visually differentiate a cherry** "
        "**leaf that is healthy from one that contains powdery mildew**\n"
        "* *This relates to data exploration and visualisation*\n"
    )
    st.info(
        "Technical Implications:"
    )
    st.write(
        "   - Mean-average and standard deviation representations to assess variability of images "
        "will be displayed for both infected and uninfected leaf classes (healthy or powdery mildew).\n"
        "   - The differences between an average infected leaf image and an uninfected leaf image will "
        "be displayed and defined.\n"
        "   - An image montage of both classes (infected and uninfected) will be collated for a clear "
        "visual representation of each class."
    )
    st.subheader("2.\n")
    st.warning(
        "**The client is interested in predicting if a cherry leaf is healthy or contains powdery mildew.**\n"
        "* *This is a binary classification problem and a suitable use case for a Convolutional Neural Network*\n"
    )
    st.info('Technical Implications:')
    st.write(
        "   - We aim to predict if a given leaf is infected or not judging by the presence of powdery mildew.\n"
        "   - We aim to use the CNN to map relationships between features and labels.\n"
        "   - We aim to build a binary classifier and generate reports."
    )
    st.write('---')