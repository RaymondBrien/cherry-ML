import streamlit as st

from typing import Union, List, Literal
from src.data_management import load_pkl_file


def load_evaluation(version: int, dataset: Union[Literal['train', 'validation', 'test'], List[str]]):
    """
    Dynamically load the evaluation pickle file for specified 
    Return a dataframe
    
    Args:
        - version: specify model version for correct output folder access
        - dataset: specify which evaluation pkl
    """
    # if a single dataset is provided as a string, convert it to a list for consistency
    if isinstance(dataset, str):
        dataset = [dataset]
    
    # dictionary to store each dataset's evaluation
    evaluations = {}
    
    for ds in dataset:
        evaluations[ds] = load_pkl_file(f'outputs/{version}/{ds}-evaluation.pkl')
    
    return evaluations

