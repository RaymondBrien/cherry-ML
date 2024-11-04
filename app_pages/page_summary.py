import streamlit as st


def page_summary_body():

    st.write("### Brief Project Summary")


    st.info(
        # TODO edit line breaks
        "**What is Mildew?**\n"
        "Mildew refers to a white powdery fungal growth that can affect the leaves of plants, including crops. It is a common plant disease that can reduce the health and productivity of affected plants.\n\n"
        "**Project Terms and Jargon:**\n"
        "- **Healthy** leaves are those without any visible signs of powdery mildew.\n"
        "- **Unhealthy** leaves are those that show white powdery mildew on its surface, characteristic of mildew infection.\n"
        "\nThe goal of this project is to build a system that can accurately detect the presence of powdery mildew on leaf images, in order to help identify and manage the client's business effectively."
    )

    # from README file - "Business Requirements" section
    st.success(
        "The project has 2 business requirements:\n"
        "1. Data Visualisation\n"
        "   - Mean-average and standard deviation representations to assess variability of images will be displayed for both infected and uninfected leaf classes (healthy or powdery mildew).\n"
        "   - The differences between an average infected leaf image and an uninfected leaf image will be displayed and defined.\n"
        "   - An image montage of both classes (infected and uninfected) will be collated for a clear visual representation of each class.\n"
        "2. Binary Classification using Convolutional Neural Networks\n"
        "   - We aim to predict if a given leaf is infected or not judging by the presence of powdery mildew.\n"
        "   - We aim to use the CNN to map relationships between features and labels.\n"
        "   - We aim to build a binary classifier and generate reports."
    )

    # link README file, for user access to full project documentation
    st.write(
        f"* For additional information, please visit and **read** the "
        f"[Project README file](https://github.com/RaymondBrien/cherry-ml).")
