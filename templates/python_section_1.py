from typing import Dict, List

import pandas as pd
import re
import numpy as np

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
    result= []
    for i in range(0, len(lst), n):
        item = lst[i : i+n]
        for j in range(len(item) - 1, -1, -1):
            result.append(item[j])
    return result

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    dict = {}
    for i in range(len(lst)):
        key = len(lst[i])
        if key not in dict:
            dict[key] = []
        dict[key].append(lst[i])
    return dict

def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here
    def flatten(current_dict, parent_key = '') -> Dict:
        items = []
        if isinstance(current_dict, dict):
            for k, v in current_dict.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                items.extend(flatten(v, new_key).items())
        elif isinstance(current_dict, list):
            for i, v in enumerate(current_dict):
                list_key = f"{parent_key}[{i}]"
                items.extend(flatten(v, list_key).items())
        else:
            items.append((parent_key, current_dict))
        return dict(items)
    return flatten(nested_dict)

def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    permutation = []
    used = [False] * len(nums)
    nums.sort()  

    def backtrack(current_permutation):
        if len(current_permutation) == len(nums):
            permutation.append(current_permutation[:])
            return

        for i in range(len(nums)):
            if used[i] or (i > 0 and nums[i] == nums[i - 1] and not used[i - 1]):
                continue
            used[i] = True
            current_permutation.append(nums[i]) 
            backtrack(current_permutation)
            used[i] = False
            current_permutation.pop()
    backtrack([])
    return permutation

def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    date_patterns = [
        r"\d{2}-\d{2}-\d{4}",  
        r"\d{2}/\d{2}/\d{4}",  
        r"\d{4}\.\d{2}\.\d{2}",  
    ]
    dates = []
    for pattern in date_patterns:
        for match in re.findall(pattern, text):
            dates.append(match)
    return dates

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """
    return pd.Dataframe()


def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here
    n = len(matrix)
    rotated_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n-1-i] = matrix[i][j]
    result_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):           
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]            
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            result_matrix[i][j] = row_sum + col_sum
            
    return result_matrix

def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here

    return pd.Series()
