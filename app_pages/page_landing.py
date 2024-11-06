import streamlit as st

def landing_page():
    st.title('Mildew Detector')
    st.subheader('*Welcome to the Mildew Detector Project*\n')
    st.write(
        '\nAccess the menu items from the side pannel '
        'to view project breakdowns, dataset visuals and project requirements.\n'
        )

    st.info(
        'If you would like to see further details, please view the project '
        'readme available here: \n'
        '\n[README](https://github.com/RaymondBrien/cherry-ml)'
    )
        