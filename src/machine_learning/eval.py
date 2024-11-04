import streamlit as st
from src.data_management import load_pkl_file


def load_test_evaluation(version):  #TODO add val and train too for sake of seeing whole picture if more info required
    """
    This function loads the test evaluation output or results
    """
    return load_pkl_file(f'outputs/{version}/test-evaluation.pkl')
