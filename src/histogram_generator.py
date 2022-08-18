import numpy as np
import pandas as pd
from pandas import DataFrame
import pickle
import os
from typing import Dict, List, Union, Tuple

def generate_histogram(data: Dict, return_pd: bool = True) -> Union[List, DataFrame]:
    keys = data.keys()

    histogram = [data[key]['distance'] for key in keys]

    if not return_pd:
        return histogram
    
    return DataFrame(histogram, columns=["distances"])

def generate_histogram_per_sequence(data: Dict) -> Dict:
    unique_paths = set()
    keys = data.keys()

    for key in keys:
        if "\\".join(str(os.path.dirname(data[key]['paths'][0])).split("\\")[-4::]) not in unique_paths:
            unique_paths.add("\\".join(str(os.path.dirname(data[key]['paths'][0])).split("\\")[-4::]))

    sequence_histograms = {}
    for u_p in unique_paths:
        list_of_points = list()
        for key in keys:
            if u_p in data[key]["paths"][0]:
                list_of_points.append(data[key]["distance"])
        sequence_histograms[f"{u_p}"] = list_of_points

    return sequence_histograms


def find_sequence(data: Dict, search_type: str) -> Tuple[str, float]:
    sequence_key = str()
    
    if search_type != "min" and search_type != "max":
        print(f"Incorrect search type {search_type}")
        print("Please use 'min' to find minimal value or 'max' to find maximum value in the dataset")
        return

    elif search_type == "max":
        sequence_avg_distance = 0.0
        for key in data.keys():
            avg = sum(data[key])/len(data[key])

            if sequence_avg_distance < avg:
                sequence_avg_distance = avg
                sequence_key = key
        
        return sequence_key, sequence_avg_distance

    else:
        sequence_avg_distance = 9999.0
        for key in data.keys():
            avg = sum(data[key])/len(data[key])

            if sequence_avg_distance > avg:
                sequence_avg_distance = avg
                sequence_key = key
        
        return sequence_key, sequence_avg_distance


def find_subsequence(data: Dict, match_path: str, search_type: str) -> Tuple[List[str], float]:
    # find subsequence with highest/lowest distance
    
    keys = data.keys()
    subsequences = list()
    distances = list()

    result_subsequence = list()
    result_distance = 0.0

    if search_type != "min" and search_type != "max":
        print(f"Incorrect search type {search_type}.")
        print("Please use 'min' to find minimal value or 'max' to find maximum value in the dataset")
        return

    for key in keys:
        if match_path in os.path.dirname(data[key]['paths'][0]):
            subsequences.append(data[key]['paths'])
            distances.append(data[key]['distance'])

    if search_type == "max":
        result_distance = max(distances)
        result_index = distances.index(result_distance)
        result_subsequence = subsequences[result_index]
    
    else:
        result_distance = min(distances)
        result_index = distances.index(result_distance)
        result_subsequence = subsequences[result_index]

    return result_subsequence, result_distance