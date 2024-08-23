import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

def stratified_sample(df, col, n_samples, task):
    """
    Performs stratified random sampling on a DataFrame.
    Parameters:
    - df: The input DataFrame.
    - col: The column to use for stratification.
    - n_samples: The number of samples to select.
    - random_state: The seed used by the random number generator for reproducibility (default: 42).
    Returns:
    - sampled_df: The stratified sampled DataFrame.
    """
    if task == "multitask":
        df[col] = df[col].apply(tuple)
    # Create a StratifiedShuffleSplit object
    sss = StratifiedShuffleSplit(n_splits=1, test_size=n_samples)
    # Get the unique values and their counts in the specified column
    value_counts = df[col].value_counts()
    # Calculate the number of samples to select from each group
    n_samples_per_group = {
        value: int(count * n_samples / len(df))
        for value, count in value_counts.items()
    }
    # Initialize an empty list to store the sampled indices
    sampled_indices = []
    # Perform stratified sampling for each group
    for value, count in n_samples_per_group.items():
        group_indices = df[df[col] == value].index
        for train_index, test_index in sss.split(group_indices, df.loc[group_indices, col]):
            sampled_indices.extend(group_indices[test_index])
    # Select the sampled rows from the DataFrame
    sampled_df = df.loc[sampled_indices]
    return sampled_df
