import os
import numpy as np
import math
from scipy.stats import wasserstein_distance
from PIL import Image
import pickle
from pathlib import Path
import copy
import cv2 as cv
from typing import List, Union
from collections import deque
import pandas as pd

# calculate the kl divergence (is not symmetrical) requires non-zero inputs in distributions
def kl_divergence(p, q):
    return sum(p[i] * math.log2(p[i]/q[i]) for i in range(len(p)))
 
# calculate the js divergence (is symmetrical)
def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

SMALL_CONST = 1e-10
FEATURE_GENERATORS = [wasserstein_distance, kl_divergence, js_divergence]
FEATURE_GENERATORS = {generator.__name__: generator for generator in FEATURE_GENERATORS}

#TODO: Add exception handling
def generate_histogram(path: str) -> np.ndarray:
    """
    Generates array of histograms of pixel frequency for a given input image.

    Args:
        path (str): Path to the image.

    Returns:
        np.ndarray: Three dimensional array containing pixel value frequencies
        for separate channels in RGB format.
    """    
    image = cv.imread(path)

    bins = 256 # Equal to the number of possible pixel values
    histRange = (0, 256)
    accumulate = False
    
    n_channels = image.shape[2]

    if not ((n_channels == 1) or (n_channels == 3)):
        raise Exception(f"Incorrect number of channels ({n_channels}) in image. Single or triple channel image required.")


    # Grayscale image case
    if n_channels == 1:
        hist = np.ravel(cv.calcHist(image, [0], None, [bins], histRange, accumulate=accumulate))
        return hist/hist.sum()


    # Color image case
    bgr_planes = cv.split(image)
    
    # Calculating histograms for separate channels
    b_hist = np.ravel(cv.calcHist(bgr_planes, [0], None, [bins], histRange, accumulate=accumulate))
    g_hist = np.ravel(cv.calcHist(bgr_planes, [1], None, [bins], histRange, accumulate=accumulate))
    r_hist = np.ravel(cv.calcHist(bgr_planes, [2], None, [bins], histRange, accumulate=accumulate))
    
    # Converting counts to frequencies
    # Adding small const to avoid problem with log() calculation for zeros
    b_hist = (b_hist/b_hist.sum()) + SMALL_CONST
    g_hist = (g_hist/g_hist.sum()) + SMALL_CONST
    r_hist = (r_hist/r_hist.sum()) + SMALL_CONST
        
    return np.array([r_hist, g_hist, b_hist])


#TODO: Modify to calculate k histograms only once, and then store values?
def generate_features(list_of_image_paths: List[str], subsequence_length: int, 
                      feature_type: Union[str, List[str]]) -> pd.DataFrame:
    """
    Given list of image paths calculates average values of chosen feature for subsequences of length n.
    Available features: "wasserstein_distance", "kl_divergence", "js_divergence".

    Args:
        list_of_image_paths (List[str]): List of paths to the full video sequence
        subsequence_length (int): Size of sliding window used in calculation of average dynamic within given sequence
        feature_type (Union[str, List[str]]): "wasserstein_distance", "kl_divergence" or "js_divergence"

    Raises:
        Exception: Subsequence length has to be greater than one
        Exception: Unknown feature type

    Returns:
        pd.DataFrame: DataFrame that containing most important informations about each sample such as:
        dataset subset (train or test); sequence type (fake or real); video number based on formatted path;
        sequence number within given video based on formatted path; first frame of the subsequence;
        length of subsequence and calculated values of desired features.

    """    
    if not subsequence_length > 1:
        raise Exception("Subsequence length has to be greater than 1")

    if isinstance(feature_type, str):
        feature_type = [feature_type]

    if not set(feature_type).issubset(set(FEATURE_GENERATORS.keys())):
        raise Exception(f"Unknown feature types {list(set(feature_type) - set(FEATURE_GENERATORS.keys()))}.")

    

    # Initialize datastructures
    histograms_cache = deque()
    distances = list()
    img_indices = deque() # Keeps track of images in sliding window

    # Collect basic data about input path
    # Requires predefined path in 
    # ".../SAMPLE_TYPE-DATASET_SUBSET/VIDEO_NUMBER/SAMPLE_TYPE/SUBSEQUENCE_NUMBER/image.png" format
    sequence = list_of_image_paths[0].split("/")[-2]
    sample_type = list_of_image_paths[0].split("/")[-3]
    video = list_of_image_paths[0].split("/")[-4]
    dataset_subset = list_of_image_paths[0].split("/")[-5].split("_")[1]

    for i in range(len(list_of_image_paths) - subsequence_length + 1):
        # On first loop generate and cache histograms for further re-use
        if len(histograms_cache) == 0:
            for k in range(subsequence_length):
                img_indices.append(int(list_of_image_paths[i+k].split("/")[-1].replace(".png", "")))
                histograms_cache.append(generate_histogram(list_of_image_paths[i+k]))

        # Generate required histograms after initial loop
        if len(histograms_cache) < subsequence_length:
            img_indices.append(int(list_of_image_paths[i+subsequence_length-1].split("/")[-1].replace(".png", "")))
            histograms_cache.append(generate_histogram(list_of_image_paths[i+subsequence_length-1]))
        
        # Prepare dictionary for current subsequence
        first_frame = img_indices.popleft()  # Remove current first image from the queue 
        sample_data = {"subset": dataset_subset,
                       "type": sample_type,
                       "video": video,
                       "sequence": sequence,
                       "first_frame": first_frame,
                       "subsequence_length": subsequence_length
        }

        for feature in feature_type:
            subsequence_average_feature = 0.0

            # Calculate average feature value for next pair of images
            for i in range(subsequence_length - 1):
                average_feature = 0.0
                n_channels = histograms_cache[0].shape[0]

                # Calculate average feature value for corresponding channels
                for j in range(n_channels):
                    average_feature += (FEATURE_GENERATORS[feature](histograms_cache[i][j], histograms_cache[i+1][j]))/n_channels
                
                subsequence_average_feature += average_feature/(subsequence_length-1)
        
            sample_data[feature] = subsequence_average_feature
        
        distances.append(pd.Series(sample_data))
        histograms_cache.popleft() # Remove leftmost histogram from queue

    return pd.DataFrame(distances)


# Save result for given seq as: seq_data-seq_len.npy
def save_avg_distance_seq(data_path: str, save_path: str, sequence_length: int, feature_type: Union[str, List[str]]):
    split_seq_path = Path(data_path).parts[::-1]
    split_seq_path = split_seq_path[0:4]
    split_seq_path = split_seq_path[::-1]

    for part in split_seq_path:
        save_path = save_path + part + "/"
    
    os.makedirs(save_path, exist_ok = True)
    
    list_of_image_paths = os.listdir(data_path)
    list_of_image_paths = sorted([int(x.replace('.png', '')) for x in list_of_image_paths])
    list_of_image_paths = [data_path.replace("\\", "/") + str(img) + ".png" for img in list_of_image_paths]
    
    distances = generate_features(list_of_image_paths, sequence_length, feature_type)

    with open(save_path + f"subsequence_data-{sequence_length}.pkl", 'wb') as f:
        pickle.dump(distances, f)
