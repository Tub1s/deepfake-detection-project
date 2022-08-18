from msilib import sequence
import pickle
import glob
import os
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Union
from collections.abc import Callable
import features.paths

def get_list_of_data_paths(main_paths: List[str]) -> List:
    paths = list()
    
    for main_path in main_paths:
        sub_paths = glob.glob(main_path + "*\\")
        paths += sub_paths

    return paths


def generate_full_dataset(aggregated_paths: List[str], idx_file_name: str, data_file_name: str) -> Dict:
    result_dict = dict()
    data_index = 0

    for path in aggregated_paths:
        with open(path+idx_file_name, 'rb') as f:
            idxs = pickle.load(f)

        with open(path+data_file_name, 'rb') as f:
            distance = np.load(f)
        
        for key in idxs.keys():
            temp = {'paths': idxs[key], 'distance': distance[key]}
            result_dict[data_index] = temp
            data_index += 1
    
        data_index += 1

    return result_dict


def dataset_sort_function(tup):
    key, d = tup
    return d["distance"]


def sort_dataset(dataset: Dict, sort_function: Callable, reverse=False) -> Dict:
    if not reverse:
        return sorted(dataset.items(), key = sort_function)

    else: 
        return list(reversed(sorted(dataset.items(), key = sort_function)))


def generate_final_dataset(full_dataset: List[Tuple[int, Dict]], starting_index: int, return_paths_only: bool, set_type: str = None):
    unique_paths = set()
    dataset = list()

    if set_type == "avg": #! Hardcoded for Q1 and Q3 test dataset; Refactoring required for automatic calculation of Q1 and Q3
        for item in full_dataset[starting_index:]:
            if (str(os.path.dirname(item[1]['paths'][0])) not in unique_paths) and (item[1]['distance'] >= 0.000105 and item[1]['distance'] <= 0.000179):
                unique_paths.add(str(os.path.dirname(item[1]['paths'][0])))
                dataset.append(item[1])
    
    else:
        for item in full_dataset[starting_index:]:
            if (str(os.path.dirname(item[1]['paths'][0])) not in unique_paths):
                unique_paths.add(str(os.path.dirname(item[1]['paths'][0])))
                dataset.append(item[1])

    if not return_paths_only:
        return dataset

    else:
        return [data['paths'] for data in dataset]


def dataset_generator(global_path: Union[str, List[str]], save_path: str, sequence_length: int, reverse: bool, paths_only: bool, set_type: str=None):
    idx_file = f"idx_frames-{sequence_length}.npy"
    data_file = f"seq_data-{sequence_length}.npy"
    

    if type(global_path) == list:
        for path in global_path:
            main_paths = glob.glob(path)
            paths = get_list_of_data_paths(main_paths)
            full_dataset = generate_full_dataset(paths, idx_file, data_file)
            sorted_full_dataset = sort_dataset(full_dataset, dataset_sort_function, reverse)
            final_dataset = generate_final_dataset(sorted_full_dataset, 0, paths_only, set_type=set_type)

            dataset_type = "highest" if reverse == True else "lowest"
            dataset_type = dataset_type if set_type != "avg" else "average"
            dataset_subtype =  "paths_only" if paths_only else "with_distances"
            save_info = os.path.dirname(path).split("\\")[-3].split("_")
            with open(f'{save_path}{save_info[1]}_{save_info[0]}_{dataset_type}_{dataset_subtype}_seq{sequence_length}_dataset.pkl', 'wb') as f:
                pickle.dump(final_dataset, f)
            
            with open(f'{save_path}{save_info[1]}_{save_info[0]}_{dataset_type}_seq{sequence_length}_full.pkl', 'wb') as f:
                pickle.dump(full_dataset, f)


if __name__ == "__main__":
    global_paths = features.paths.WILDDEEPFAKE_DATASET
    save_path = features.paths.SAVED_DATASETS

    #False + "avg"

    # dataset_generator(global_paths, save_path, 5, False, True) #Lowest, paths_only
    # dataset_generator(global_paths, save_path, 5, True, True) #Highest, paths_only
    # dataset_generator(global_paths, save_path, 5, False, False) #Lowest, with_distances
    # dataset_generator(global_paths, save_path, 5, True, False) #Highest, with_distances
    dataset_generator(global_paths, save_path, 10, False, True, "avg") #Avg, paths_only
    dataset_generator(global_paths, save_path, 10, False, False, "avg") #Avg, with_distances
