import streamlit as st
import numpy as np
import pandas as pd

from PIL import Image
from src.data_management import download_dataframe_as_csv
from src.machine_learning.predictive_analysis import (
    load_model_and_predict,
    resize_input_image,
    plot_predictions_probabilities
)


def page_mildew_detector_body():
    """
    Display mildew detector page body
    Display dataset source url
    Allow user image upload for live predictions
    Holds 'make prediction' button
    """

    st.header("Powdery Mildew Detection")
    st.success(
        f"**Business Requirement 2**\n"
        f" The client is interested in predicting whether a cherry leaf"
        f" contains powdery mildew or not.")

    st.write('---')

    st.info(
        "* Download a set of leaves containing both powdery mildew"
        "or healthy leaves for live prediction.\n"
        "Images can be downloaded from "
        "[**here**](https://www.kaggle.com/codeinstitute/cherry-leaves)\n"
    )
    st.write(
        f"**Upload a clear picture of a cherry leaf for live predictions. You may select more than one.**"
    )
    images_buffer = st.file_uploader(' ',
                                     accept_multiple_files=True)

    st.write(
        f" Click **Make Prediction** button for prediction results"
    )

    predict_button = st.button("Make Prediction")
    if predict_button:
        make_live_predict(images_buffer)


def make_live_predict(images_buffer):
    """
    Open uploaded user image
    Log size
    Resize image
    Load model and predict
    Report prediction results analysis report
    """

    if images_buffer is not None:
        df_report = pd.DataFrame([])
        for image in images_buffer:

            img_pil = (Image.open(image))
            st.info(f"Cherry leaves Sample: **{image.name}**")
            img_array = np.array(img_pil)
            st.image(
                img_pil,
                caption=f"Image Size: {img_array.shape[1]}px width x {img_array.shape[0]}px height")

            version = 'v6'  # TODO update as needed

            resized_img = resize_input_image(img=img_pil, version=version)

            pred_proba, pred_class = load_model_and_predict(
                resized_img, version=version)

            plot_predictions_probabilities(pred_proba, pred_class)

            df_report = df_report.append(
                {"Name": image.name, 'Result': pred_class}, ignore_index=True)

        if not df_report.empty:
            st.success("Analysis Report")
            st.table(df_report)
            st.markdown(
                download_dataframe_as_csv(df_report),
                unsafe_allow_html=True)