import os
import itertools
import random
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.image import imread


def page_leaf_visualiser_body():
    """ Create page body for leaf visualisation image montages"""

    st.header("Cherry Leaves Visualiser")

    st.success(
        "As part of the dataset exploration phase, this page provides a reference "
        "of the visual differences between **healthy** Cherry Leaves "
        "and Cherry Leaves infected with **powdery mildew**, in answer to "
        "Business Requirement No.1 as defined by the client.\n"
    )
    version = 'v6'  # TODO confirm version
    if st.checkbox("Differences between average and variability images"):

        avg_var_healty = plt.imread(f"outputs/{version}/avg_var_healthy.png")
        avg_var_powdery_mildew = plt.imread(
            f"outputs/{version}/avg_var_powdery_mildew.png")
        st.success(
            f" A significant visual difference in consistent colouring has been \n"
            f" observed: \n"
            f" We notice that the average and variability images of infected leaves \n"
            f" have more white color blotches in their surface than a healthy leaf \n"
            f" which presents more greenish uniform coloring.\n"
            f" However the average and variability images did not show any clear patterns \n"
            f" that could intuitively differentiate infected leaves and healthy leaves."
        )
        st.image(avg_var_healty, caption='Healty leaves - Avegare and Variability')
        st.image(avg_var_powdery_mildew,
                 caption='Powdery Mildew infected leaves - Average and Variability')
        st.write("---")
    # Show the difference between average and variability images
    if st.checkbox("Differences between average healthy and average infected leaves"):
        diff_between_avgs = plt.imread(f"outputs/{version}/avg_diff.png")
        st.success(
            f"* We appreciate similar pattern here, where the healthy leaves have a "
            f"more clear, green surface, and the infected ones have more white color in "
            f"the surface. But again it is difficult to compare both "
            f" kind of leaves."
        )

        st.image(diff_between_avgs, caption='Difference between average images')

    # Show the image montage
    if st.checkbox("Image Montage"):
        st.write(
            "* To create and refresh the montage, click on 'Create Montage' button")
        st.info(
            f"* The montage helps to visually differentiate between a healthy"
            f" leaf and an infected one. The infected one has white, powdery "
            f"spots or patches on the top side of leaves"
        )
        my_data_dir = 'inputs/cherryleaves_dataset/cherry-leaves'
        labels = os.listdir(my_data_dir + '/validation')
        label_to_display = st.selectbox(
            label="Select label:", options=labels, index=0)
        if st.button("Create Montage"):
            image_montage(dir_path=my_data_dir + '/validation',
                          label_to_display=label_to_display,
                          nrows=8, ncols=3, figsize=(10, 25))
        st.write("---")


def image_montage(dir_path, label_to_display, nrows, ncols, figsize=(15, 10)):
    """


    """
    sns.set_style("dark")
    labels = os.listdir(dir_path)

    # checks if your montage space is greater than subset size
    images_list = os.listdir(dir_path+'/' + label_to_display)
    if nrows * ncols < len(images_list):
        img_idx = random.sample(images_list, nrows * ncols)
    else:
        print(
            f"Decrease nrows or ncols to create your montage. \n"
            f"There are {len(images_list)} in your subset. "
            f"You requested a montage with {nrows * ncols} spaces")
        return

    # create list of axes indices based on nrows and ncols
    list_rows = range(0, nrows)
    list_cols = range(0, ncols)
    plot_idx = list(itertools.product(list_rows, list_cols))

    # create figure and display images
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for x in range(0, nrows*ncols):
        img = imread(f"{dir_path}/{label_to_display}/{img_idx[x]}")
        img_shape = img.shape
        axes[plot_idx[x][0], plot_idx[x][1]].imshow(img)
        axes[plot_idx[x][0], plot_idx[x][1]].set_title(
            f"Width {img_shape[1]}px x Height {img_shape[0]}px")
        axes[plot_idx[x][0], plot_idx[x][1]].set_xticks([])
        axes[plot_idx[x][0], plot_idx[x][1]].set_yticks([])
    plt.tight_layout()

    st.pyplot(fig=fig)


    # Display image montage of label_to_display,
    # selected from checkbox.