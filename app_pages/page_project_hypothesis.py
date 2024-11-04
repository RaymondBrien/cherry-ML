import streamlit as st


def project_hypothesis_page_body():
    """Displays project hypothesis and validation page content."""

    st.header("Project Hypothesis and Validation")

    st.info(

        "Hypothesis 1: It is hypothesized that infected cherry tree leaves will be visually "
        "differentiable from healthy cherry tree leaves. Specifically, white powdery "
        "mildew will be present on the surface of the majority of infected leaves "
        "within the dataset.\n"

        # TODO update to correct accuracy metric
        "* Validation 1: Based on the data visualization notebook, infected leaves displayed "
        "white powdery mildew on their surfaces across multiple image montages, "
        "making them visually distinct from healthy leaves. Although generating a "
        "mathematical difference between average images of each class did not "
        "reveal strongly visible pattern differences to the human eye, the "
        "the class variability images showed sufficient shape contrasts within the raw "
        "dataset to train a CNN successfully.\n"
        "Furthermore, the trained ML model achieves an accuracy of x% in distinguishing "
        "between the two classes, providing a quantitative metric to support "
        "this hypothesis.\n"

        "Hypothesis 2: It is hypothesised that using only the provided dataset, the ML model will "
        "be able to distinguish between a healthy cherry leaf and an infected cherry leaf "
        "with at least 97% accuracy."
        # TODO update to correct accuracy metric
        "* Validation 2: The model can able to distinguish between image classes correctly with "
        "x% accuracy, certified by the evaluation metrics"

        "Hypothesis 3: Augmenting the training dataset with transformations (like rotation, zoom, "
        "or contrast changes) will improve the model's performance."
        # TODO validte this with images
        "* Validation 3: "
            )
