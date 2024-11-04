import streamlit as st
from src.data_management import load_pkl_file


def load_test_evaluation(version):  #TODO add val and train too?
    """
    This function loads the test evaluation output or results
    """
    return load_pkl_file(f'outputs/{version}/evaluation.pkl')
