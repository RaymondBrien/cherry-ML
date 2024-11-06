import streamlit as st


def project_hypothesis_page_body():
    """Displays project hypothesis and validation page content."""

    st.header("Project Hypothesis and Validation")
    st.info('**Hypothesis 1**')
    st.markdown("""
    - **Hypothesis 1**: It is hypothesized that infected cherry tree leaves will be visually differentiable from healthy cherry tree leaves. Specifically, white powdery mildew will be present on the surface of
    the majority of infected leaves within the dataset.
    - **Validation**:
        - **Image Montage**: To visually illustrate the differences between healthy and infected leaves.
        - **Average Image Per Class**: Calculating the average image for both healthy and infected leaves to identify any distinct color or texture patterns.
        - **Difference Between Averages**: Comparing the average images of healthy and infected leaves to highlight visual differences.
    """)

    st.info('**Hypothesis 2**')
    st.markdown("""
        - **Hypotheesis 2**: It is hypothesised that using only the provided dataset, the ML model will be able to distinguish between a healthy cherry leaf and an infected cherry leaf with at least 97% accuracy.
        - **Validation**:
            - **Accuracy Evaluation on Test Set**: After model training, model evaluation will measure accuracy and loss readings. Accuracy above 97% per business requirement 2, will pass. This is ultimately the main business driver.
            - **Model Output**: Accuracy 99.88% (*Pass True*)
    """)

    st.info('**Hypothesis 3**')
    st.markdown("""
        - **Hypothesis 3 (technical)**: It is hypothesised that at least 20 epochs will be required for effective generalisation of the data from the given dataset.
        - **Validation**: Epoch count during training once model has been fitted
            - **Model Output**: From early stopping, only 11 epochs were needed for suitable accuracy results.
    """)




